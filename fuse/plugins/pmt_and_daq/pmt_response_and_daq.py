from immutabledict import immutabledict
import straxen 
import strax
from numba import njit
import numpy as np
import logging

from scipy.interpolate import interp1d

export, __all__ = strax.exporter()

from ...common import FUSE_PLUGIN_TIMEOUT

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.pmt_and_daq.pmt_response_and_daq')

@export
class PMTResponseAndDAQ(strax.Plugin):
    
    __version__ = "0.0.0"

    depends_on = ("photon_summary", "pulse_ids", "pulse_windows")

    provides = 'raw_records'
    data_kind = 'raw_records'

    dtype = strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH)

    save_when = strax.SaveWhen.ALWAYS

    input_timeout = FUSE_PLUGIN_TIMEOUT

    #Config options
    debug = straxen.URLConfig(
        default=False, type=bool,track=False,
        help='Show debug informations',
    )

    deterministic_seed = straxen.URLConfig(
        default=True, type=bool,
        help='Set the random seed from lineage and run_id, or pull the seed from the OS.',
    )

    dt = straxen.URLConfig(
        type=(int),
        help='sample_duration',
    )

    pmt_circuit_load_resistor = straxen.URLConfig(
        help='pmt_circuit_load_resistor', type=(int, float),
    )

    external_amplification = straxen.URLConfig(
        type=(int, float),
        help='external_amplification',
    )

    digitizer_bits = straxen.URLConfig(
        type=(int, float),
        help='digitizer_bits',
    )

    digitizer_voltage_range = straxen.URLConfig(
        type=(int, float),
        help='digitizer_voltage_range',
    )

    noise_data = straxen.URLConfig(
        cache=True,
        help='noise_data',
    )

    pe_pulse_ts = straxen.URLConfig(
        help='pe_pulse_ts',
    )
    
    pe_pulse_ys = straxen.URLConfig(
        help='pe_pulse_ys',
    )

    pmt_pulse_time_rounding = straxen.URLConfig(
        type=(int, float),
        help='pmt_pulse_time_rounding',
    )

    samples_after_pulse_center = straxen.URLConfig(
        type=(int, float),
        help='samples_after_pulse_center',
    )

    samples_before_pulse_center = straxen.URLConfig(
        type=(int, float),
        help='samples_before_pulse_center',
    )

    digitizer_reference_baseline = straxen.URLConfig(
        type=(int, float),
        help='digitizer_reference_baseline',
    )

    zle_threshold = straxen.URLConfig(
        type=(int, float),
        help='zle_threshold',
    )

    trigger_window = straxen.URLConfig(
        type=(int, float),
        help='trigger_window',
    )

    samples_to_store_before = straxen.URLConfig(
        type=(int, float),
        help='samples_to_store_before',
    )

    def setup(self):

        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running PMTResponseAndDAQ in debug mode")
        else: 
            log.setLevel('WARNING')

        if self.deterministic_seed:
            hash_string = strax.deterministic_hash((self.run_id, self.lineage))
            seed = int(hash_string.encode().hex(), 16)
            self.rng = np.random.default_rng(seed = seed)
            log.debug(f"Generating random numbers from seed {seed}")
        else: 
            self.rng = np.random.default_rng()
            log.debug(f"Generating random numbers with seed pulled from OS")

        self.current_2_adc = self.pmt_circuit_load_resistor \
                * self.external_amplification \
                / (self.digitizer_voltage_range / 2 ** (self.digitizer_bits))

        #Lets use normal distributed noise for now. 
        #Check later if we need to add "real measured" noise
        self.noise_mean = np.mean(self.noise_data['arr_0'], axis=0)
        self.noise_std = np.std(self.noise_data['arr_0'], axis=0)

        self._pmt_current_templates, _template_length = self.init_pmt_current_templates()

        self.threshold = self.digitizer_reference_baseline - self.zle_threshold - 1
        self.pulse_left_extension = + int(self.samples_to_store_before)+ self.samples_before_pulse_center

    def compute(self, propagated_photons, pulse_windows):

        if len(propagated_photons) == 0 or len(pulse_windows) == 0:
            return np.zeros(0, dtype=self.dtype)
        
        #use an upper limit for the waveform buffer
        length_waveform_buffer = np.int32(np.sum(np.ceil(pulse_windows["length"]/strax.DEFAULT_RECORD_LENGTH)))
        waveform_buffer = np.zeros(length_waveform_buffer, dtype = self.dtype)

        propagated_photons = propagated_photons[propagated_photons["pulse_id"].argsort()]
        index_values = np.unique(propagated_photons["pulse_id"], return_index=True)[1]

        buffer_level = build_waveform(
            pulse_windows,
            propagated_photons,
            index_values,
            waveform_buffer,
            self.pulse_left_extension,
            self.dt,
            self._pmt_current_templates,
            self.current_2_adc,
            self.noise_mean,
            self.noise_std,
            self.digitizer_reference_baseline,
            self.rng,
            self.threshold,
            self.trigger_window
            )

        records = waveform_buffer[:buffer_level] 

        #Digitzier saturation
        #Clip negative values to 0
        records['data'][records['data']<0] = 0   

        records = strax.sort_by_time(records)

        return records

    def init_pmt_current_templates(self):
        """
        Create spe templates, for 10ns sample duration and 1ns rounding we have:
        _pmt_current_templates[i] : photon timing fall between [10*m+i, 10*m+i+1)
        (i, m are integers)
        """

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

        return _pmt_current_templates, _template_length


@njit()
def build_waveform(
    photon_pulses,
    photons,
    index_values,
    waveform_buffer,
    pulse_left_extension,
    dt, 
    pmt_current_templates,
    current_2_adc,
    noise_mean,
    noise_std,
    digitizer_reference_baseline,
    rng,
    threshold,
    trigger_window,
    ):
    
    buffer_level = 0
    start = index_values
    stop = np.zeros(len(index_values), dtype = np.int64)
    stop[:-1] = index_values[1:]
    stop[-1] = len(photons)

    #Iterate over all pulses
    # for pulse, photons_to_put_into_pulse in zip(photon_pulses, photons_per_pulse):
    for i, pulse in enumerate(photon_pulses):

        pulse_length = pulse["length"] + pulse_left_extension
        pulse_waveform_buffer = np.zeros(pulse_length)

        photons_to_put_into_pulse = photons[start[i]:stop[i]]
            
        add_current(photons_to_put_into_pulse['time'],
                    photons_to_put_into_pulse['photon_gain'],
                    pulse["time"]//dt-pulse_left_extension,
                    dt,
                    pmt_current_templates,
                    pulse_waveform_buffer)
        
        pulse_waveform_buffer = - pulse_waveform_buffer * current_2_adc
        #Add normal distributed noise
        noise = np.around(rng.normal(noise_mean[pulse["channel"]],
                                        noise_std[pulse["channel"]],
                                        size = pulse_length,
                                        )).astype(np.int64).T

        pulse_waveform_buffer += noise

        add_baseline(pulse_waveform_buffer, digitizer_reference_baseline)

        buffer_level = convert_pulse_to_fragments(
            pulse_waveform_buffer,
            waveform_buffer,
            buffer_level,
            threshold,
            trigger_window,
            pulse["channel"],
            pulse["time"],
            dt,
            pulse_left_extension
            )

    return buffer_level


@njit()
def convert_pulse_to_fragments(
    single_waveform,
    waveform_buffer,
    buffer_level,
    threshold,
    trigger_window,
    pulse_channel,
    pulse_time,
    dt,
    pulse_left_extension
    ): 

    zle_intervals_buffer = -1 * np.ones((np.int64(len(single_waveform)/2), 2), dtype=np.int64)

    n_itvs_found = find_intervals_below_threshold(
        single_waveform,
        threshold,
        trigger_window + trigger_window + 1,
        zle_intervals_buffer
        )
    
    itvs_to_encode = zle_intervals_buffer[:n_itvs_found]
    itvs_to_encode[:, 0] -= trigger_window
    itvs_to_encode[:, 1] += trigger_window
    itvs_to_encode = np.clip(itvs_to_encode, 0, len(single_waveform) - 1)
    # Land trigger window on even numbers
    itvs_to_encode[:, 0] = np.ceil(itvs_to_encode[:, 0] / 2.0) * 2
    itvs_to_encode[:, 1] = np.floor(itvs_to_encode[:, 1] / 2.0) * 2

    #can we get rid of this loop?
    for interval in itvs_to_encode:

        waveform_to_encode = single_waveform[interval[0]:interval[1]+1]

        pulse_length = interval[1] - interval[0] + 1
        waveform_split, records_needed = split_data(waveform_to_encode, strax.DEFAULT_RECORD_LENGTH)

        s = slice(buffer_level, buffer_level + records_needed)
        waveform_buffer['channel'][s] = pulse_channel
        waveform_buffer['data'][s] = waveform_split
        waveform_buffer['time'][s] =  dt * (pulse_time//dt + interval[0] + strax.DEFAULT_RECORD_LENGTH * np.arange(records_needed)) - 10 * pulse_left_extension
        waveform_buffer['dt'][s] = dt
        waveform_buffer['pulse_length'][s] = pulse_length
        waveform_buffer['record_i'][s] = np.arange(records_needed)
        waveform_buffer['length'][s] = [min(pulse_length, strax.DEFAULT_RECORD_LENGTH * (i+1))
                                            - strax.DEFAULT_RECORD_LENGTH * i for i in range(records_needed)]

        buffer_level += records_needed
        
    return buffer_level

@njit
def add_baseline(data, baseline):
    data += baseline

@njit()
def split_data(data, samples_per_record):
    """
    Split data into arrays of length samples_per_record and pad with zeros if necessary
    """
    data_length = len(data)
    arrays_needed = int(np.ceil(data_length / samples_per_record))
    pad_array = np.zeros(arrays_needed * samples_per_record)
    pad_array[0:len(data)] = data
    sliced_data = pad_array.reshape((-1, samples_per_record))
    return sliced_data, arrays_needed

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


@njit()
def find_intervals_below_threshold(w, threshold, holdoff, result_buffer):
    """Fills result_buffer with l, r bounds of intervals in w < threshold.
    :param w: Waveform to do hitfinding in
    :param threshold: Threshold for including an interval
    :param holdoff: Holdoff number of samples after the pulse return back down to threshold
    :param result_buffer: numpy N*2 array of ints, will be filled by function.
                          if more than N intervals are found, none past the first N will be processed.
    :returns : number of intervals processed
    Boundary indices are inclusive, i.e. the right boundary is the last index which was < threshold
    """
    result_buffer_size = len(result_buffer)
    last_index_in_w = len(w) - 1

    in_interval = False
    current_interval = 0
    current_interval_start = -1
    current_interval_end = -1

    for i, x in enumerate(w):

        if x < threshold:
            if not in_interval:
                # Start of an interval
                in_interval = True
                current_interval_start = i

            current_interval_end = i

        if ((i == last_index_in_w and in_interval) or
                (x >= threshold and i >= current_interval_end + holdoff and in_interval)):
            # End of the current interval
            in_interval = False

            # Add bounds to result buffer
            result_buffer[current_interval, 0] = current_interval_start
            result_buffer[current_interval, 1] = current_interval_end
            current_interval += 1

            if current_interval == result_buffer_size:
                result_buffer[current_interval, 1] = len(w) - 1

    n_intervals = current_interval  # No +1, as current_interval was incremented also when the last interval closed
    return n_intervals