from immutabledict import immutabledict
import straxen 
import strax
import os
from numba import njit
import numpy as np
import logging

from strax import deterministic_hash
from scipy.interpolate import interp1d

from ..common import find_intervals_below_threshold

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('XeSim.pmt_and_daq.pmt_response_and_daq')
log.setLevel('WARNING')

private_files_path = "path/to/private/files"
config = straxen.get_resource(os.path.join(private_files_path, 'sim_files/fax_config_nt_sr0_v4.json') , fmt='json')

@strax.takes_config(
    strax.Option('rext', default=100000, track=False, infer_type=False,
                 help="right raw extension"),
    strax.Option('pmt_transit_time_spread', default=config['pmt_transit_time_spread'], track=False, infer_type=False,
                 help="pmt_transit_time_spread"),
    strax.Option('dt', default=config['sample_duration'], track=False, infer_type=False,
                 help="sample_duration"),
    strax.Option('to_pe_file', default=os.path.join(private_files_path,"sim_files/to_pe_nt.npy"), track=False, infer_type=False,
                 help="to_pe file"),
    strax.Option('digitizer_voltage_range', default=config['digitizer_voltage_range'], track=False, infer_type=False,
                 help="digitizer_voltage_range"),
    strax.Option('digitizer_bits', default=config['digitizer_bits'], track=False, infer_type=False,
                 help="digitizer_bits"),
    strax.Option('pmt_circuit_load_resistor', default=config['pmt_circuit_load_resistor'], track=False, infer_type=False,
                 help="pmt_circuit_load_resistor"),
    strax.Option('photon_area_distribution', default=config['photon_area_distribution'], track=False, infer_type=False,
                 help="photon_area_distribution"),
    strax.Option('samples_to_store_before', default=config['samples_to_store_before'], track=False, infer_type=False,
                 help="samples_to_store_before"),
    strax.Option('samples_before_pulse_center', default=config['samples_before_pulse_center'], track=False, infer_type=False,
                 help="samples_before_pulse_center"),
    strax.Option('samples_to_store_after', default=config['samples_to_store_after'], track=False, infer_type=False,
                 help="samples_to_store_after"),
    strax.Option('samples_after_pulse_center', default=config['samples_after_pulse_center'], track=False, infer_type=False,
                 help="samples_after_pulse_center"),
    strax.Option('pmt_pulse_time_rounding', default=config['pmt_pulse_time_rounding'], track=False, infer_type=False,
                 help="pmt_pulse_time_rounding"),
    strax.Option('external_amplification', default=config['external_amplification'], track=False, infer_type=False,
                 help="external_amplification"),
    strax.Option('trigger_window', default=config['trigger_window'], track=False, infer_type=False,
                 help="trigger_window"),
    strax.Option('n_top_pmts', default=253, track=False, infer_type=False,
                 help="n_top_pmts"),
    strax.Option('n_tpc_pmts', default=494, track=False, infer_type=False,
                 help="n_tpc_pmts"),
    strax.Option('detector', default="XENONnT", track=False, infer_type=False,
                 help="detector"),
    strax.Option('high_energy_deamplification_factor', default=config['high_energy_deamplification_factor'], track=False, infer_type=False,
                 help="high_energy_deamplification_factor"),
    strax.Option('enable_noise', default=config['enable_noise'], track=False, infer_type=False,
                 help="enable_noise"),
    strax.Option('digitizer_reference_baseline', default=config['digitizer_reference_baseline'], track=False, infer_type=False,
                 help="digitizer_reference_baseline"),
    strax.Option('zle_threshold', default=config['zle_threshold'], track=False, infer_type=False,
                 help="zle_threshold"),
    strax.Option('debug', default=False, track=False, infer_type=False,
                 help="Show debug informations"),
)
class pmt_response_and_daq(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ("photon_channels_and_timeing", "S1_channel_and_timings", "pmt_afterpulses")
    
    provides = ('raw_records', 'raw_records_he', 'raw_records_aqmon')#, 'truth')
    data_kind = immutabledict(zip(provides, provides))
    
    def setup(self):

        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running pmt_response_and_daq in debug mode")
        
        
        to_pe = straxen.get_resource(self.to_pe_file, fmt='npy')
        self.to_pe = to_pe[0][1]

        adc_2_current = (self.digitizer_voltage_range
                / 2 ** (self.digitizer_bits)
                 / self.pmt_circuit_load_resistor)

        self.gains = np.divide(adc_2_current,
                              self.to_pe,
                              out=np.zeros_like(self.to_pe),
                              where=self.to_pe != 0)

        self.pmt_mask = np.array(self.gains) > 0  # Converted from to pe (from cmt by default)
        self.turned_off_pmts = np.arange(len(self.gains))[np.array(self.gains) == 0]
        
        self.photon_area_distribution = straxen.get_resource(self.photon_area_distribution, fmt='csv')
        
        
        self.pe_pulse_ts = config["pe_pulse_ts"]
        self.pe_pulse_ys = config["pe_pulse_ys"]
        
        
        #setup for Digitize Pulse Cache 
        self.current_2_adc = self.pmt_circuit_load_resistor \
                * self.external_amplification \
                / (self.digitizer_voltage_range / 2 ** (self.digitizer_bits))
        
        self.channels_bottom = np.arange(self.n_top_pmts, self.n_tpc_pmts)
        #self.channel_map = dict(st.config['channel_map'])
        self.channel_map = {'tpc': (0, 493),
                            'he': (500, 752),
                            'aqmon': (790, 807),
                            'aqmon_nv': (808, 815),
                            'tpc_blank': (999, 999),
                            'mv': (1000, 1083),
                            'aux_mv': (1084, 1087),
                            'mv_blank': (1999, 1999),
                            'nveto': (2000, 2119),
                            'nveto_blank': (2999, 2999)
                           } #I can get this from the context...but how??
        
        
        self.channel_map['sum_signal'] = 800
        self.channels_bottom = np.arange(self.n_top_pmts, self.n_tpc_pmts)
        
        if self.enable_noise:
            self.noise_data = straxen.get_resource(config['noise_file'], fmt='npy')['arr_0']
             
        #ZLE Part (Now building actual raw_records data!)
        self.samples_per_record = strax.DEFAULT_RECORD_LENGTH
        #self.blevel = 0  # buffer_filled_level
        self.record_buffer = np.zeros(5000000, dtype=strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH))
        
        self.special_thresholds = config.get('special_thresholds', {})

        #Im sure there is a better way to handle this part
        self._cached_pmt_current_templates = {}
        
    def compute(self, S1_photons, S2_photons, AP_photons):

        if len(S1_photons) == 0 and len(S2_photons) == 0  and len(AP_photons) == 0:
            return dict(raw_records=np.zeros(0, dtype=strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH)),
                        raw_records_he=np.zeros(0, dtype=strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH)),
                        raw_records_aqmon=np.zeros(0, dtype=strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH)))
        
        merged_photons = np.concatenate([S1_photons, S2_photons, AP_photons])
        S1_photons = None
        S2_photons = None
        AP_photons = None
        
        #Sort all photons by time
        sortind = np.argsort(merged_photons["time"])
        merged_photons = merged_photons[sortind]
        
        #Split the photons into groups that will be simualted at once
        split_photons = np.split(merged_photons, np.where(np.diff(merged_photons["time"]) > self.rext)[0]+1)
        
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
                        ch_he = np.arange(self.channel_map['he'][0],
                                          self.channel_map['he'][1] + 1)[ch]
                        _raw_data[ch_he, _slice] += adc_wave_he
                        _channel_mask[ch_he] = True
                        _channel_mask['left'][ch_he] = _channel_mask['left'][ch]
                        _channel_mask['right'][ch_he] = _channel_mask['right'][ch]
                    elif ch <= self.channels_bottom[-1]:
                        sum_signal(adc_wave_he,
                                   _pulse['left'] - left,
                                   _pulse['right'] - left + 1,
                                   _raw_data[self.channel_map['sum_signal']])
                        
            _channel_mask['left'] -= left + config['trigger_window']
            _channel_mask['right'] -= left - config['trigger_window']
            
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
        
        return dict(raw_records=records[records['channel'] < self.channel_map['he'][0]],
                       raw_records_he=records[(records['channel'] >= self.channel_map['he'][0]) &
                                              (records['channel'] <= self.channel_map['he'][-1])],
                       raw_records_aqmon=records[records['channel'] == 800],
                       #truth=_truth
                    ) 
        
    def init_pmt_current_templates(self):
        """
        Create spe templates, for 10ns sample duration and 1ns rounding we have:
        _pmt_current_templates[i] : photon timing fall between [10*m+i, 10*m+i+1)
        (i, m are integers)
        """
        h = deterministic_hash(config)
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