from immutabledict import immutabledict
import straxen 
import strax
import os
from numba import njit
import numpy as np
import logging

from strax import deterministic_hash
from scipy.interpolate import interp1d

export, __all__ = strax.exporter()

from ...common import find_intervals_below_threshold

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.pmt_and_daq.pmt_response_and_daq')
log.setLevel('WARNING')

@export
class PMTResponseAndDAQ(strax.Plugin):
    
    __version__ = "0.0.0"

    depends_on = ("photon_summary")
    
    provides = ('raw_records', 'raw_records_he', 'raw_records_aqmon')#, 'truth')
    data_kind = immutabledict(zip(provides, provides))
    
    #Config options
    debug = straxen.URLConfig(
        default=False, type=bool,
        help='Show debug informations',
    )

    zle_threshold = straxen.URLConfig(
        type=(int, float),
        help='zle_threshold',
    )

    digitizer_reference_baseline = straxen.URLConfig(
        type=(int, float),
        help='digitizer_reference_baseline',
    )

    enable_noise = straxen.URLConfig(
        type=bool,
        help='enable_noise',
    )

    high_energy_deamplification_factor = straxen.URLConfig(
        type=(int, float),
        help='high_energy_deamplification_factor',
    )

    detector = straxen.URLConfig(
        help='Detector to be simulated',
    )

    n_top_pmts = straxen.URLConfig(
        type=(int),
        help='Number of PMTs on top array',
    )

    n_tpc_pmts = straxen.URLConfig(
        type=(int),
        help='Number of PMTs in the TPC',
    )

    trigger_window = straxen.URLConfig(
        type=(int, float),
        help='trigger_window',
    )

    external_amplification = straxen.URLConfig(
        type=(int, float),
        help='external_amplification',
    )

    pmt_pulse_time_rounding = straxen.URLConfig(
        type=(int, float),
        help='pmt_pulse_time_rounding',
    )

    samples_after_pulse_center = straxen.URLConfig(
        type=(int, float),
        help='samples_after_pulse_center',
    )

    samples_to_store_after = straxen.URLConfig(
        type=(int, float),
        help='samples_to_store_after',
    )

    samples_before_pulse_center = straxen.URLConfig(
        type=(int, float),
        help='samples_before_pulse_center',
    )

    samples_to_store_before = straxen.URLConfig(
        type=(int, float),
        help='samples_to_store_before',
    )

    pmt_circuit_load_resistor = straxen.URLConfig(
        help='pmt_circuit_load_resistor', type=(int, float),
    )

    digitizer_bits = straxen.URLConfig(
        type=(int, float),
        help='digitizer_bits',
    )

    digitizer_voltage_range = straxen.URLConfig(
        type=(int, float),
        help='digitizer_voltage_range',
    )

    dt = straxen.URLConfig(
        type=(int),
        help='sample_duration',
    )

    pmt_transit_time_spread = straxen.URLConfig(
        type=(int, float),
        help='pmt_transit_time_spread',
    )

    rext = straxen.URLConfig(
        type=(int),
        help='right raw extension',
    )
    
    pe_pulse_ts = straxen.URLConfig(
        help='pe_pulse_ts',
    )
    
    pe_pulse_ys = straxen.URLConfig(
        help='pe_pulse_ys',
    )
    
    gains = straxen.URLConfig(
        cache=True,
        help='pmt gains',
    )
    
    photon_area_distribution = straxen.URLConfig(
        cache=True,
        help='photon_area_distribution',
    )
    
    special_thresholds = straxen.URLConfig(
        help='special_thresholds',
    )
    
    noise_data_tmp = straxen.URLConfig(
        cache=True,
        help='noise_data',
    )
    
    channel_map = straxen.URLConfig(
        track=False, type=immutabledict,
        help="immutabledict mapping subdetector to (min, max) "
             "channel number.")

    def setup(self):

        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running pmt_response_and_daq in debug mode")

        self.pmt_mask = np.array(self.gains) > 0  # Converted from to pe (from cmt by default)
        self.turned_off_pmts = np.arange(len(self.gains))[np.array(self.gains) == 0]

        
        #setup for Digitize Pulse Cache 
        self.current_2_adc = self.pmt_circuit_load_resistor \
                * self.external_amplification \
                / (self.digitizer_voltage_range / 2 ** (self.digitizer_bits))
        
        self.channels_bottom = np.arange(self.n_top_pmts, self.n_tpc_pmts)

        #Ugly hack below
        self.channel_map_mutable = dict(self.channel_map)
        self.channel_map_mutable['sum_signal'] = 800
        
        self.channels_bottom = np.arange(self.n_top_pmts, self.n_tpc_pmts)
        
        #We can most likely get rid of this hack and make it properly....
        if self.enable_noise:
            self.noise_data = self.noise_data_tmp['arr_0']
             
        #ZLE Part (Now building actual raw_records data!)
        self.samples_per_record = strax.DEFAULT_RECORD_LENGTH
        #self.blevel = 0  # buffer_filled_level
        self.record_buffer = np.zeros(5000000, dtype=strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH))

        #Im sure there is a better way to handle this part
        self._cached_pmt_current_templates = {}
        
    def compute(self, propagated_photons):

        if len(propagated_photons) == 0:
            return dict(raw_records=np.zeros(0, dtype=strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH)),
                        raw_records_he=np.zeros(0, dtype=strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH)),
                        raw_records_aqmon=np.zeros(0, dtype=strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH)))
        
        
        #Split the photons into groups that will be simualted at once
        split_photons = np.split(propagated_photons, np.where(np.diff(propagated_photons["time"]) > self.rext)[0]+1)
        
        self.blevel = 0  # buffer_filled_level
        
        #Start the first loop.... so many loops in this plugin :-(
        for photon_group in split_photons:
            
            sort_idx = np.argsort(photon_group["channel"])
            _photon_timings = photon_group[sort_idx]["time"]
            _photon_channels = photon_group[sort_idx]["channel"]
            _photon_gains = photon_group[sort_idx]["photon_gain"]
            
            _pulses = []
            
            _pmt_current_templates, _template_length = self.init_pmt_current_templates()
            
            #Hey! Now there is a loop in a loop! 
            counts_start = 0  # Secondary loop index for assigning channel
            for channel, counts in zip(*np.unique(_photon_channels, return_counts=True)):
                # Use 'counts' amount of photon for this channel
                _channel_photon_timings = _photon_timings[counts_start:counts_start+counts]
                _channel_photon_gains = _photon_gains[counts_start:counts_start+counts]

                counts_start += counts
                if channel in self.turned_off_pmts:
                    continue

                #skip truth here

                # Build a simulated waveform, length depends on min and max of photon timings
                min_timing, max_timing = np.min(
                    _channel_photon_timings), np.max(_channel_photon_timings)

                pulse_left = (int(min_timing // self.dt)
                              - int(self.samples_to_store_before)
                              - self.samples_before_pulse_center)

                pulse_right = (int(max_timing // self.dt)
                               + int(self.samples_to_store_after)
                               + self.samples_after_pulse_center)
                pulse_current = np.zeros(pulse_right - pulse_left + 1)


                add_current(_channel_photon_timings.astype(np.int64),
                            _channel_photon_gains,
                             pulse_left,
                             self.dt,
                            _pmt_current_templates,
                            pulse_current
                           )

                _pulses.append(dict(photons  = len(_channel_photon_timings),
                                    channel  = channel,
                                    left     = pulse_left,
                                    right    = pulse_right,
                                    duration = pulse_right - pulse_left + 1,
                                    current  = pulse_current,
                                   )
                              )
                
            #The code from Digitize Pulse Cache
            left = np.min([p['left'] for p in _pulses]) - self.trigger_window
            right = np.max([p['right'] for p in _pulses]) + self.trigger_window
            pulse_length = right - left
            
            assert right - left < 1000000, "Pulse cache too long"

            if left % 2 != 0:
                left -= 1  # Seems like a digizier effect

            _raw_data = np.zeros((801, right - left + 1), dtype='<i8')
            
            # Use this mask to by pass non-activated channels
            # Set to true when working with real noise
            _channel_mask = np.zeros(801, dtype=[('mask', '?'), ('left', 'i8'), ('right', 'i8')])
            _channel_mask['left'] = int(2**63-1)
            
            #Time for another loop!
            for _pulse in _pulses:
                ch = _pulse['channel']
                _channel_mask['mask'][ch] = True
                _channel_mask['left'][ch] = min(_pulse['left'], _channel_mask['left'][ch])
                _channel_mask['right'][ch] = max(_pulse['right'], _channel_mask['right'][ch])
                adc_wave = - np.around(_pulse['current'] * self.current_2_adc).astype(np.int64)
                _slice = slice(_pulse['left'] - left, _pulse['right'] - left + 1)

                _raw_data[ch, _slice] += adc_wave

                if self.detector == 'XENONnT':
                    adc_wave_he = adc_wave * int(self.high_energy_deamplification_factor)
                    if ch < self.n_top_pmts:
                        ch_he = np.arange(self.channel_map_mutable['he'][0],
                                          self.channel_map_mutable['he'][1] + 1)[ch]
                        _raw_data[ch_he, _slice] += adc_wave_he
                        _channel_mask[ch_he] = True
                        _channel_mask['left'][ch_he] = _channel_mask['left'][ch]
                        _channel_mask['right'][ch_he] = _channel_mask['right'][ch]
                    elif ch <= self.channels_bottom[-1]:
                        sum_signal(adc_wave_he,
                                   _pulse['left'] - left,
                                   _pulse['right'] - left + 1,
                                   _raw_data[self.channel_map_mutable['sum_signal']])
                        
            _channel_mask['left'] -= left + self.trigger_window
            _channel_mask['right'] -= left - self.trigger_window
            
            # Adding noise, baseline and digitizer saturation
            if self.enable_noise:
                add_noise(data=_raw_data,
                          channel_mask=_channel_mask,
                          noise_data=self.noise_data,
                          noise_data_length=len(self.noise_data),
                          noise_data_channels=len(self.noise_data[0]))
                
                
            add_baseline(_raw_data, _channel_mask, 
                  self.digitizer_reference_baseline,)
            
            digitizer_saturation(_raw_data, _channel_mask)
            
            #ZLE part
            zle_intervals_buffer = -1 * np.ones((50000, 2), dtype=np.int64)
            
            #LOOPS! WE NEED MORE LOOPS!
            for ix, data in enumerate(_raw_data):
                if not _channel_mask['mask'][ix]:
                    continue
                channel_left, channel_right = _channel_mask['left'][ix], _channel_mask['right'][ix]
                data = data[channel_left:channel_right+1]

                # For simulated data taking reference baseline as baseline
                # Operating directly on digitized downward waveform        
                if str(ix) in self.special_thresholds:
                    threshold = self.digitizer_reference_baseline \
                        - self.special_thresholds[str(ix)] - 1
                else:
                    threshold = self.digitizer_reference_baseline - self.zle_threshold - 1

                n_itvs_found = find_intervals_below_threshold(
                    data,
                    threshold=threshold,
                    holdoff=self.trigger_window + self.trigger_window + 1,
                    result_buffer=zle_intervals_buffer,)

                itvs_to_encode = zle_intervals_buffer[:n_itvs_found]
                itvs_to_encode[:, 0] -= self.trigger_window
                itvs_to_encode[:, 1] += self.trigger_window
                itvs_to_encode = np.clip(itvs_to_encode, 0, len(data) - 1)
                # Land trigger window on even numbers
                itvs_to_encode[:, 0] = np.ceil(itvs_to_encode[:, 0] / 2.0) * 2
                itvs_to_encode[:, 1] = np.floor(itvs_to_encode[:, 1] / 2.0) * 2
                
                #LOOOOOOOOPS!
                for itv in itvs_to_encode:

                    #This happens in chunk_raw_records
                    channel = ix
                    left_tmp = left + channel_left + itv[0]
                    right_tmp = left + channel_left + itv[1]
                    data_tmp = data[itv[0]:itv[1]+1]

                    pulse_length = right_tmp - left_tmp + 1
                    records_needed = int(np.ceil(pulse_length / self.samples_per_record))
                    
                    #Leave out the if cases for now

                    # WARNING baseline and area fields are zeros before finish_results
                    s = slice(self.blevel, self.blevel + records_needed)
                    
                    self.record_buffer[s]['channel'] = channel
                    self.record_buffer[s]['dt'] = self.dt
                    self.record_buffer[s]['time'] = self.dt * (left_tmp + self.samples_per_record * np.arange(records_needed))
                    self.record_buffer[s]['length'] = [min(pulse_length, self.samples_per_record * (i+1))
                                                  - self.samples_per_record * i for i in range(records_needed)]
                    self.record_buffer[s]['pulse_length'] = pulse_length
                    self.record_buffer[s]['record_i'] = np.arange(records_needed)
                    self.record_buffer[s]['data'] = np.pad(data_tmp,
                                                           (0, records_needed * self.samples_per_record - pulse_length),
                                                           'constant').reshape((-1, self.samples_per_record))
                    self.blevel += records_needed
         
        #We made it through all the loops! 
        records = self.record_buffer[:self.blevel] 
        records = strax.sort_by_time(records)
        
        return dict(raw_records=records[records['channel'] < self.channel_map_mutable['he'][0]],
                       raw_records_he=records[(records['channel'] >= self.channel_map_mutable['he'][0]) &
                                              (records['channel'] <= self.channel_map_mutable['he'][-1])],
                       raw_records_aqmon=records[records['channel'] == 800],
                       #truth=_truth
                    ) 
        
    #Check if we can just move this function into the setup method!!!
    def init_pmt_current_templates(self):
        """
        Create spe templates, for 10ns sample duration and 1ns rounding we have:
        _pmt_current_templates[i] : photon timing fall between [10*m+i, 10*m+i+1)
        (i, m are integers)
        """
        h = deterministic_hash("Placeholder!")
        #if h in self._cached_pmt_current_templates:
        #    _pmt_current_templates = self._cached_pmt_current_templates[h]
        #    return _pmt_current_templates

        # Interpolate on cdf ensures that each spe pulse would sum up to 1 pe*sample duration^-1
        pe_pulse_function = interp1d(self.pe_pulse_ts,
                                     np.cumsum(self.pe_pulse_ys),
                                     bounds_error=False, fill_value=(0, 1)
                                    )

        # Samples are always multiples of sample_duration
        sample_duration = self.dt
        samples_before = self.samples_before_pulse_center
        samples_after = self.samples_after_pulse_center
        pmt_pulse_time_rounding = self.pmt_pulse_time_rounding

        # Let's fix this, so everything can be turned into int
        assert pmt_pulse_time_rounding == 1

        samples = np.linspace(-samples_before * sample_duration,
                              + samples_after * sample_duration,
                              1 + samples_before + samples_after)
        _template_length = np.int64(len(samples) - 1)

        templates = []
        for r in np.arange(0, sample_duration, pmt_pulse_time_rounding):
            pmt_current = np.diff(pe_pulse_function(samples - r)) / sample_duration  # pe / 10 ns
            # Normalize here to counter tiny rounding error from interpolation
            pmt_current *= (1 / sample_duration) / np.sum(pmt_current)  # pe / 10 ns
            templates.append(pmt_current)
        _pmt_current_templates = np.array(templates)
        self._cached_pmt_current_templates[h] = _pmt_current_templates

        return _pmt_current_templates, _template_length
    
    def infer_dtype(self):
        dtype = {data_type: strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH)
                 for data_type in self.provides if data_type != 'truth'}

        #dtype['truth'] = instruction_dtype + self._truth_dtype
        return dtype
    
    
@njit
def add_current(photon_timings,
                photon_gains,
                pulse_left,
                dt,
                pmt_current_templates,
                pulse_current):
    #         """
    #         Simulate single channel waveform given the photon timings
    #         photon_timing         - dim-1 integer array of photon timings in unit of ns
    #         photon_gain           - dim-1 float array of ph. 2 el. gain individual photons
    #         pulse_left            - left of the pulse in unit of 10 ns
    #         dt                    - mostly it is 10 ns
    #         pmt_current_templates - list of spe templates of different reminders
    #         pulse_current         - waveform
    #         """
    if not len(photon_timings):
        return
    
    template_length = len(pmt_current_templates[0])
    i_photons = np.argsort(photon_timings)
    # Convert photon_timings to int outside this function
    # photon_timings = photon_timings // 1

    gain_total = 0
    tmp_photon_timing = photon_timings[i_photons[0]]
    for i in i_photons:
        if photon_timings[i] > tmp_photon_timing:
            start = int(tmp_photon_timing // dt) - pulse_left
            reminder = int(tmp_photon_timing % dt)
            pulse_current[start:start + template_length] += \
                pmt_current_templates[reminder] * gain_total

            gain_total = photon_gains[i]
            tmp_photon_timing = photon_timings[i]
        else:
            gain_total += photon_gains[i]

    start = int(tmp_photon_timing // dt) - pulse_left
    reminder = int(tmp_photon_timing % dt)
    pulse_current[start:start + template_length] += \
        pmt_current_templates[reminder] * gain_total
    
@njit
def sum_signal(adc_wave, left, right, sum_template):
    sum_template[left:right] += adc_wave
    return sum_template

@njit
def add_noise(data, channel_mask, noise_data, noise_data_length, noise_data_channels):
    """
    Get chunk(s) of noise sample from real noise data
    """
    if channel_mask['mask'].sum() == 0:
        return

    left = np.min(channel_mask['left'][channel_mask['mask']])
    right = np.max(channel_mask['right'][channel_mask['mask']])

    if noise_data_length-right+left-1 < 0:
        high = noise_data_length-1
    else:
        high = noise_data_length-right+left-1
    if high <= 0:
        ix_rand = 0
    else:
        ix_rand = np.random.randint(low=0, high=high)

    for ch in range(data.shape[0]):
        # In case adding noise to he channels is not supported
        if ch >= noise_data_channels:
            continue

        if not channel_mask['mask'][ch]:
            continue

        left, right = channel_mask['left'][ch], channel_mask['right'][ch]
        for ix_data in range(left, right+1):
            ix_noise = ix_rand + ix_data - left
            if ix_data >= len(data[ch]):
                # Don't create value-errors
                continue

            if ix_noise >= noise_data_length:
                ix_noise -= noise_data_length * (ix_noise // noise_data_length)

            data[ch, ix_data] += noise_data[ix_noise, ch]
            
            
@njit
def add_baseline(data, channel_mask, baseline):
    for ch in range(data.shape[0]):
        if not channel_mask['mask'][ch]:
            continue
        left, right = channel_mask['left'][ch], channel_mask['right'][ch]
        for ix in range(left, right+1):
            data[ch, ix] += baseline
            
            
@njit
def digitizer_saturation(data, channel_mask):
    for ch in range(data.shape[0]):
        if not channel_mask['mask'][ch]:
            continue
        left, right = channel_mask['left'][ch], channel_mask['right'][ch]
        for ix in range(left, right+1):
            if data[ch, ix] < 0:
                data[ch, ix] = 0