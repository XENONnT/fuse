import numpy as np
from numba import njit
from scipy.interpolate import interp1d
import strax
import straxen

from ...plugin import FuseBaseDownChunkingPlugin
from ...common import stable_sort, stable_argsort

export, __all__ = strax.exporter()


@export
class PMTResponseAndDAQ(FuseBaseDownChunkingPlugin):
    """Plugin to simulate the PMT response and DAQ effects.

    First the single PMT waveform is simulated based on the photon
    timing and gain information. Next the waveform is converted to ADC
    counts, noise and a baseline are added. Then hitfinding is performed
    and the found intervals are split into multiple fragments of fixed
    length (if needed). Finally the data is saved as raw_records.
    """

    __version__ = "0.1.6"

    depends_on = ("photon_summary", "pulse_ids", "pulse_windows")

    provides = "raw_records"
    data_kind = "raw_records"

    dtype = strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH)

    save_when = strax.SaveWhen.TARGET

    # Config options
    dt = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=sample_duration",
        type=int,
        cache=True,
        help="Width of one sample [ns]",
    )

    pmt_circuit_load_resistor = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=pmt_circuit_load_resistor",
        type=(int, float),
        cache=True,
        help="PMT circuit load resistor [kg m^2/(s^3 A)]",
    )

    external_amplification = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=external_amplification",
        type=(int, float),
        cache=True,
        help="External amplification factor",
    )

    digitizer_bits = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?&fmt=json&take=digitizer_bits",
        type=(int, float),
        cache=True,
        help="Number of bits of the digitizer boards",
    )

    digitizer_voltage_range = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=digitizer_voltage_range",
        type=(int, float),
        cache=True,
        help="Voltage range of the digitizer boards [V]",
    )

    noise_data = straxen.URLConfig(
        default="simple_load://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=noise_file"
        "&fmt=npy",
        cache=True,
        help="Measured noise data",
    )

    enable_noise = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?&fmt=json&take=enable_noise",
        cache=True,
        help="Option to enable or disable noise",
    )

    pe_pulse_ts = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?&fmt=json&take=pe_pulse_ts",
        cache=True,
        help="Time for PMT SPE waveform [sample]",
    )

    pe_pulse_ys = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?&fmt=json&take=pe_pulse_ys",
        cache=True,
        help="Amplitude for PMT SPE waveform [PE/sample]",
    )

    pmt_pulse_time_rounding = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=pmt_pulse_time_rounding",
        type=(int, float),
        cache=True,
        help="Time rounding of the PMT pulse",
    )

    samples_after_pulse_center = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=samples_after_pulse_center",
        type=(int, float),
        cache=True,
        help="Number of samples after the pulse center",
    )

    samples_before_pulse_center = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=samples_before_pulse_center",
        type=(int, float),
        cache=True,
        help="Number of samples before the pulse center",
    )

    digitizer_reference_baseline = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=digitizer_reference_baseline",
        type=(int, float),
        cache=True,
        help="Digitizer reference baseline",
    )

    zle_threshold = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?&fmt=json&take=zle_threshold",
        type=(int, float),
        cache=True,
        help="Threshold for the zero length encoding",
    )

    trigger_window = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?&fmt=json&take=trigger_window",
        type=(int, float),
        cache=True,
        help="Trigger window",
    )

    special_thresholds = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=special_thresholds",
        cache=True,
        help="Special thresholds for certain PMTs",
    )

    n_tpc_pmts = straxen.URLConfig(
        type=int,
        help="Number of PMTs in the TPC",
    )

    raw_records_file_size_target = straxen.URLConfig(
        type=(int, float),
        default=200,
        track=False,
        help="Target for the raw records file size [MB]",
    )

    min_records_gap_length_for_splitting = straxen.URLConfig(
        type=(int, float),
        default=1e5,
        track=False,
        help="chunk can not be split if gap between pulses is smaller than this value given in ns",
    )

    def setup(self):
        super().setup()

        self.current_2_adc = (
            self.pmt_circuit_load_resistor
            * self.external_amplification
            / (self.digitizer_voltage_range / 2 ** (self.digitizer_bits))
        )

        self._pmt_current_templates, _template_length = self.init_pmt_current_templates()

        threshold = self.digitizer_reference_baseline - self.zle_threshold - 1
        self.thresholds = threshold = np.ones(self.n_tpc_pmts) * threshold
        for key, value in self.special_thresholds.items():
            if np.int32(key) < self.n_tpc_pmts:
                self.thresholds[np.int32(key)] = self.digitizer_reference_baseline - value - 1

    def compute(self, propagated_photons, pulse_windows, start, end):
        if len(propagated_photons) == 0 or len(pulse_windows) == 0:
            self.log.debug("No photons or pulse windows found for chunk!")

            yield self.chunk(start=start, end=end, data=np.zeros(0, dtype=self.dtype))
            return  # Exit early

        # Split into "sub-chunks"
        pulse_gaps = pulse_windows["time"][1:] - strax.endtime(pulse_windows)[:-1]
        pulse_gaps = np.append(pulse_gaps, 0)  # Add 0 for last pulse gap

        split_index = find_split_index(
            pulse_windows,
            pulse_gaps,
            file_size_limit=self.raw_records_file_size_target,
            min_gap_length=self.min_records_gap_length_for_splitting,
        )

        pulse_window_chunks = np.array_split(pulse_windows, split_index)

        n_chunks = len(pulse_window_chunks)
        if n_chunks > 1:
            self.log.info(
                f"Chunk size exceeding file size target. Downchunking to {n_chunks} chunks"
            )

        photon_chunks = []
        for pulse_groups in pulse_window_chunks:
            mask = np.isin(propagated_photons["pulse_id"], pulse_groups["pulse_id"])
            photon_chunks.append(split_photons(propagated_photons[mask]))

        last_start = start
        for i, (pulse_groups, photon_chunk) in enumerate(zip(pulse_window_chunks, photon_chunks)):
            sorted_photons, photon_starts, photon_stops = photon_chunk
            records = self.compute_chunk(
                sorted_photons, photon_starts, photon_stops, pulse_groups
            )
            if i < n_chunks - 1:
                chunk_end = np.max(strax.endtime(records))
            else:
                chunk_end = end
            chunk = self.chunk(start=last_start, end=chunk_end, data=records)
            last_start = chunk_end
            yield chunk

    def compute_chunk(self, sorted_photons, photon_starts, photon_stops, pulse_groups):
        # sort pulse groups by pulse id, same order as the photons
        pulse_groups = stable_sort(pulse_groups, order="pulse_id")

        # use an upper limit for the waveform buffer
        length_waveform_buffer = np.int32(
            np.sum(np.ceil(pulse_groups["length"] / strax.DEFAULT_RECORD_LENGTH))
        )
        waveform_buffer = np.zeros(length_waveform_buffer, dtype=self.dtype)

        # Generate a random offset for noise sampling to avoid systematic biases
        # when using fixed event spacing (where events at the same time would get identical noise)
        noise_len = self.noise_data["arr_0"].shape[0]
        noise_offset = self.rng.integers(0, noise_len)

        # Hoist field views out of the @njit kernel — slicing
        # ``sorted_photons[s:e]["time"]`` inside numba is fragile across versions;
        # 1D field arrays are zero-copy views the kernel handles cleanly.
        photon_time = sorted_photons["time"]
        photon_gain = sorted_photons["photon_gain"]

        buffer_level = build_waveform(
            pulse_groups,
            photon_time,
            photon_gain,
            photon_starts,
            photon_stops,
            waveform_buffer,
            self.dt,
            self._pmt_current_templates,
            self.current_2_adc,
            self.noise_data["arr_0"].T,
            self.enable_noise,
            self.digitizer_reference_baseline,
            self.thresholds,
            self.trigger_window,
            noise_offset,
        )

        records = waveform_buffer[:buffer_level]

        # Digitzier saturation
        # Clip negative values to 0
        records["data"][records["data"] < 0] = 0
        records = strax.sort_by_time(records)

        return records

    def init_pmt_current_templates(self):
        """Create spe templates, for 10ns sample duration and 1ns rounding we
        have:

        _pmt_current_templates[i] : photon timing fall between [10*m+i,
        10*m+i+1) (i, m are integers)
        """

        # Interpolate on cdf ensures that each spe pulse would sum up to 1 pe*sample duration^-1
        pe_pulse_function = interp1d(
            self.pe_pulse_ts,
            np.cumsum(self.pe_pulse_ys) / np.sum(self.pe_pulse_ys),
            bounds_error=False,
            fill_value=(0, 1),
        )

        # Samples are always multiples of sample_duration
        sample_duration = self.dt
        samples_before = self.samples_before_pulse_center
        samples_after = self.samples_after_pulse_center
        pmt_pulse_time_rounding = self.pmt_pulse_time_rounding

        # Let's fix this, so everything can be turned into int
        assert pmt_pulse_time_rounding == 1

        samples = np.linspace(
            -samples_before * sample_duration,
            +samples_after * sample_duration,
            1 + samples_before + samples_after,
        )
        _template_length = np.int64(len(samples) - 1)

        templates = []
        for r in np.arange(0, sample_duration, pmt_pulse_time_rounding):
            pmt_current = np.diff(pe_pulse_function(samples - r)) / sample_duration  # pe / 10 ns
            # Normalize here to counter tiny rounding error from interpolation
            pmt_current *= (1 / sample_duration) / np.sum(pmt_current)  # pe / 10 ns
            templates.append(pmt_current)
        _pmt_current_templates = np.array(templates)

        return _pmt_current_templates, _template_length


@njit(cache=True)
def find_split_index(pulses, gaps, file_size_limit, min_gap_length):
    data_size_mb = 0
    split_index = []

    for i, (p, g) in enumerate(zip(pulses, gaps)):
        # Assumes data is later saved as int16
        data_size_mb += p["length"] * 2 / 1e6

        if data_size_mb < file_size_limit:
            continue

        if g >= min_gap_length:
            data_size_mb = 0
            split_index.append(i)

    return np.array(split_index) + 1


@njit(cache=True)
def build_waveform(
    pulse_windows,
    photon_time,
    photon_gain,
    photon_starts,
    photon_stops,
    waveform_buffer,
    dt,
    pmt_current_templates,
    current_2_adc,
    noise_data,
    enable_noise,
    digitizer_reference_baseline,
    thresholds,
    trigger_window,
    noise_offset=0,
):
    buffer_level = 0

    # Iterate over all pulses. (photon_time, photon_gain) are flat 1D views of
    # sorted-by-pulse_id photons; (photon_starts, photon_stops) bracket each
    # pulse_window's photons. This replaces the per-pulse_id numba.typed.List
    # the caller used to build — bit-identical and avoids the per-pulse Python
    # boxing (~26 k typed.List.append calls per K40 chunk in vanilla).
    for i in range(pulse_windows.shape[0]):
        pulse = pulse_windows[i]
        pulse_length = pulse["length"]
        pulse_waveform_buffer = np.zeros(pulse_length)

        s = photon_starts[i]
        e = photon_stops[i]
        add_current(
            photon_time[s:e],
            photon_gain[s:e],
            pulse["time"] // dt,
            dt,
            pmt_current_templates,
            pulse_waveform_buffer,
        )

        pulse_waveform_buffer = -np.around(pulse_waveform_buffer * current_2_adc).astype(np.int64)

        if enable_noise:
            # Remember to transpose the noise...
            pulse_waveform_buffer = add_noise(
                pulse_waveform_buffer, pulse["time"], noise_data[pulse["channel"]], noise_offset
            )

        add_baseline(pulse_waveform_buffer, digitizer_reference_baseline)

        buffer_level = convert_pulse_to_fragments(
            pulse_waveform_buffer,
            waveform_buffer,
            buffer_level,
            thresholds[pulse["channel"]],
            trigger_window,
            pulse["channel"],
            pulse["time"],
            dt,
        )

    return buffer_level


@njit(cache=True)
def add_noise(array, time, noise_in_channel, noise_offset=0):
    time = np.int64(time / 10)

    len_data = len(array)
    len_noise = len(noise_in_channel)

    index = (time + noise_offset + np.arange(len_data) + 1) % len_noise

    return array + noise_in_channel[index]


@njit(cache=True)
def convert_pulse_to_fragments(
    single_waveform,
    waveform_buffer,
    buffer_level,
    threshold,
    trigger_window,
    pulse_channel,
    pulse_time,
    dt,
):
    zle_intervals_buffer = -1 * np.ones((np.int64(len(single_waveform) / 2), 2), dtype=np.int64)

    n_itvs_found = find_intervals_below_threshold(
        single_waveform, threshold, trigger_window + trigger_window + 1, zle_intervals_buffer
    )

    itvs_to_encode = zle_intervals_buffer[:n_itvs_found]
    itvs_to_encode[:, 0] -= trigger_window
    itvs_to_encode[:, 1] += trigger_window
    itvs_to_encode = np.clip(itvs_to_encode, 0, len(single_waveform) - 1)
    # Land trigger window on even numbers
    itvs_to_encode[:, 0] = np.ceil(itvs_to_encode[:, 0] / 2.0) * 2
    itvs_to_encode[:, 1] = np.floor(itvs_to_encode[:, 1] / 2.0) * 2

    # Can we get rid of this loop?
    for interval in itvs_to_encode:
        waveform_to_encode = single_waveform[interval[0] : interval[1] + 1]

        pulse_length = interval[1] - interval[0] + 1
        waveform_split, records_needed = split_data(waveform_to_encode, strax.DEFAULT_RECORD_LENGTH)

        s = slice(buffer_level, buffer_level + records_needed)
        waveform_buffer["channel"][s] = pulse_channel
        waveform_buffer["data"][s] = waveform_split
        waveform_buffer["time"][s] = dt * (
            pulse_time // dt + interval[0] + strax.DEFAULT_RECORD_LENGTH * np.arange(records_needed)
        )
        waveform_buffer["dt"][s] = dt
        waveform_buffer["pulse_length"][s] = pulse_length
        waveform_buffer["record_i"][s] = np.arange(records_needed)
        waveform_buffer["length"][s] = [
            min(pulse_length, strax.DEFAULT_RECORD_LENGTH * (i + 1))
            - strax.DEFAULT_RECORD_LENGTH * i
            for i in range(records_needed)
        ]

        buffer_level += records_needed

    return buffer_level


@njit(cache=True)
def add_baseline(data, baseline):
    data += baseline


@njit(cache=True)
def split_data(data, samples_per_record):
    """Split data into arrays of length samples_per_record and pad with zeros
    if necessary."""
    data_length = len(data)
    arrays_needed = int(np.ceil(data_length / samples_per_record))
    pad_array = np.zeros(arrays_needed * samples_per_record)
    pad_array[0 : len(data)] = data
    sliced_data = pad_array.reshape((-1, samples_per_record))
    return sliced_data, arrays_needed


@njit(cache=True)
def add_current(photon_timings, photon_gains, pulse_left, dt, pmt_current_templates, pulse_current):
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
    i_photons = stable_argsort(photon_timings)
    # Convert photon_timings to int outside this function
    # photon_timings = photon_timings // 1

    gain_total = 0
    tmp_photon_timing = photon_timings[i_photons[0]]
    for i in i_photons:
        if photon_timings[i] > tmp_photon_timing:
            start = int(tmp_photon_timing // dt) - pulse_left
            reminder = int(tmp_photon_timing % dt)
            pulse_current[start : start + template_length] += (
                pmt_current_templates[reminder] * gain_total
            )

            gain_total = photon_gains[i]
            tmp_photon_timing = photon_timings[i]
        else:
            gain_total += photon_gains[i]

    start = int(tmp_photon_timing // dt) - pulse_left
    reminder = int(tmp_photon_timing % dt)
    pulse_current[start : start + template_length] += pmt_current_templates[reminder] * gain_total


@njit(cache=True)
def find_intervals_below_threshold(w, threshold, holdoff, result_buffer):
    """Fills result_buffer with l, r bounds of intervals in w < threshold.

    Args:
        w: Waveform to do hitfinding in
        threshold: Threshold for including an interval
        holdoff: Holdoff number of samples after the pulse return
            back down to threshold
        result_buffer: numpy N*2 array of ints, will be filled by
            function. if more than N intervals are found, none past the
            first N will be processed.
    Returns:
        number of intervals processed Boundary indices are inclusive,
            i.e. the right boundary is the last index which was < threshold
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

        if (i == last_index_in_w and in_interval) or (
            x >= threshold and i >= current_interval_end + holdoff and in_interval
        ):
            # End of the current interval
            in_interval = False

            # Add bounds to result buffer
            result_buffer[current_interval, 0] = current_interval_start
            result_buffer[current_interval, 1] = current_interval_end
            current_interval += 1

            if current_interval == result_buffer_size:
                result_buffer[current_interval, 1] = len(w) - 1

    # No + 1, as current_interval was incremented also when the last interval closed
    n_intervals = current_interval
    return n_intervals


@njit(cache=True, nogil=True)
def _pulse_id_group_offsets(sorted_pulse_ids, n_pulse_groups):
    """Given pulse_ids sorted ascending, return (starts, stops) per unique id.

    `starts[k]` and `stops[k]` bracket the photons whose pulse_id equals the
    k-th unique pulse_id in ``sorted_pulse_ids`` (sorted order). The caller is
    responsible for confirming that the unique-pulse-id count matches the
    pulse_groups it expects to iterate over (i.e. ``n_pulse_groups`` is the
    expected number of distinct pulse_ids).
    """
    n = sorted_pulse_ids.shape[0]
    starts = np.empty(n_pulse_groups, np.int64)
    stops = np.empty(n_pulse_groups, np.int64)
    if n == 0 or n_pulse_groups == 0:
        return starts[:0], stops[:0]
    starts[0] = 0
    g = 0
    for i in range(1, n):
        if sorted_pulse_ids[i] != sorted_pulse_ids[i - 1]:
            stops[g] = i
            g += 1
            if g < n_pulse_groups:
                starts[g] = i
    stops[g] = n
    return starts[: g + 1], stops[: g + 1]


def split_photons(propagated_photons):
    """Sort photons by pulse_id and return (sorted_array, starts, stops).

    The previous implementation returned ``list[structured_array]`` (one entry
    per unique pulse_id). That required:
      - Building a numba ``typed.List`` of structured arrays in compute_chunk
        (per-pulse Python boxing — 337 k calls/K40-chunk).
      - Calling ``np.split`` on the sorted struct array (extra reorder cost).
    The new flat-array form lets ``build_waveform`` slice the sorted photons
    directly via per-pulse (start, stop) offsets — no typed.List, no np.split.

    Bit-identical to the original modulo Python representation: same photons,
    same order within each pulse group, same number of groups.
    """
    sort_index = stable_argsort(propagated_photons["pulse_id"])
    propagated_photons_sorted = propagated_photons[sort_index]
    pulse_ids = propagated_photons_sorted["pulse_id"]

    # Count unique pulse_ids in one pass — feeds the kernel its output size.
    n = pulse_ids.shape[0]
    if n == 0:
        empty = np.empty(0, np.int64)
        return propagated_photons_sorted, empty, empty
    n_groups = 1
    for k in range(1, n):
        if pulse_ids[k] != pulse_ids[k - 1]:
            n_groups += 1
    starts, stops = _pulse_id_group_offsets(pulse_ids, n_groups)
    return propagated_photons_sorted, starts, stops
