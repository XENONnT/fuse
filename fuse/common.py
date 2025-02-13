import numpy as np
import awkward as ak
import numba

from scipy.interpolate import interp1d

# Lets wait 10 minutes for the plugin to finish
FUSE_PLUGIN_TIMEOUT = 600


kind_colors = dict(
    geant4_interactions="#40C4F3",
    clustered_interactions="#FBCE56",
    tpc_interactions="#F44E43",
    below_cathode_interactions="#F44E43",
    interactions_in_roi="#56C46C",
    s1_photons="#54E4CF",
    s2_photons="#54E4CF",
    ap_photons="#54E4CF",
    propagated_photons="#54E4CF",
    pulse_ids="#54E4CF",
    pulse_windows="#F78C27",
    raw_records="#0260EF",
    individual_electrons="#F44E43",
)


@numba.njit()
def dynamic_chunking(data, scale, n_min):
    idx_sort = np.argsort(data)
    idx_undo_sort = np.argsort(idx_sort)

    data_sorted = data[idx_sort]

    diff = data_sorted[1:] - data_sorted[:-1]

    clusters = [0]
    c = 0
    n_cluster = 0
    for value in diff:
        if value <= scale:
            clusters.append(c)
            n_cluster += 1
        elif n_cluster + 1 < n_min:
            clusters.append(c)
            n_cluster += 1
        elif value > scale:
            c = c + 1
            clusters.append(c)
            n_cluster = 0

    clusters = np.array(clusters)
    clusters_undo_sort = clusters[idx_undo_sort]

    return clusters_undo_sort


def full_array_to_numpy(array, dtype):
    len_output = len(awkward_to_flat_numpy(array["x"]))

    numpy_data = np.zeros(len_output, dtype=dtype)

    for field in array.fields:
        numpy_data[field] = awkward_to_flat_numpy(array[field])

    return numpy_data


@numba.njit()
def _sample_from_distribution(p, channel, spe_scaling_factor_distributions):
    """Function to sample from a SPE scaling factor distribution for a given
    channel."""

    indices = np.int64(p * 2000) + 1
    return spe_scaling_factor_distributions[channel, indices]


@numba.njit()
def sample_spe_scaling_factors(p, channel, spe_scaling_factor_distributions):
    """Function to sample the spe scaling factors for multiple photons."""

    result = []
    for i in range(len(p)):
        result.append(
            _sample_from_distribution(
                p[i],
                channel=channel[i],
                spe_scaling_factor_distributions=spe_scaling_factor_distributions,
            )
        )
    return np.array(result)


# Epix functions


def reshape_awkward(array, offset):
    """Function which reshapes an array of strings or numbers according to a
    list of offsets. Only works for a single jagged layer.

    Args:
        array: Flatt array which should be jagged.
        offset: Length of subintervals


    Returns:
        res: awkward1.ArrayBuilder object.
    """
    res = ak.ArrayBuilder()
    if (
        (array.dtype == np.int32)
        or (array.dtype == np.int64)
        or (array.dtype == np.float64)
        or (array.dtype == np.float32)
    ):
        _reshape_awkward_number(array, offset, res)
    else:
        _reshape_awkward_string(array, offset, res)
    return res.snapshot()


@numba.njit
def _reshape_awkward_number(array, offsets, res):
    """Function which reshapes an array of numbers according to a list of
    offsets. Only works for a single jagged layer.

    Args:
        array: Flatt array which should be jagged.
        offsets: Length of subintervals
        res: awkward1.ArrayBuilder object

    Returns:
        res: awkward1.ArrayBuilder object
    """
    start = 0
    end = 0
    for o in offsets:
        end += o
        res.begin_list()
        for value in array[start:end]:
            res.real(value)
        res.end_list()
        start = end


def _reshape_awkward_string(array, offsets, res):
    """Function which reshapes an array of strings according to a list of
    offsets. Only works for a single jagged layer.

    Args:
        array: Flatt array which should be jagged.
        offsets: Length of subintervals
        res: awkward1.ArrayBuilder object

    Returns:
        res: awkward1.ArrayBuilder object
    """
    start = 0
    end = 0
    for o in offsets:
        end += o
        res.begin_list()
        for value in array[start:end]:
            res.string(value)
        res.end_list()
        start = end


def awkward_to_flat_numpy(array):
    if len(array) == 0:
        return ak.to_numpy(array)
    return ak.to_numpy(ak.flatten(array))


def calc_dt(result):
    """Calculate dt, the time difference from the initial data in the event.

    Args:
        result: Including ``t`` field

    Returns:
        numpy.array: Array like time difference
    """
    if len(result) == 0:
        return np.empty(0)
    dt = result["t"] - result["t"][:, 0]
    return dt


def ak_num(array, **kwargs):
    """awkward.num() wrapper also for work in empty array
    Args:
        array: Data containing nested lists to count.

    Keyword Args:
        kwargs: keywords arguments for awkward.num().

    Returns:
        numpy.array: an array of integers specifying the number of elements at a
            particular level. If array is empty, return empty.
    """
    if len(array) == 0:
        return ak.from_numpy(np.empty(0, dtype="int64"))
    return ak.num(array, **kwargs)


# Code shared between S1 and S2 photon propagation
def init_spe_scaling_factor_distributions(spe_shapes):
    # Create a converter array from uniform random numbers to SPE gains
    # (one interpolator per channel)
    # Scale the distributions so that they have an SPE mean of 1 and then calculate the cdf
    uniform_to_pe_arr = []
    for ch in spe_shapes.columns[1:]:  # skip the first element which is the 'charge' header
        if spe_shapes[ch].sum() > 0:
            # mean_spe = (spe_shapes['charge'].values * spe_shapes[ch]).sum() / spe_shapes[ch].sum()
            scaled_bins = spe_shapes["charge"].values  # / mean_spe
            cdf = np.cumsum(spe_shapes[ch]) / np.sum(spe_shapes[ch])
        else:
            # If sum is 0, just make some dummy axes to pass to interpolator
            cdf = np.linspace(0, 1, 10)
            scaled_bins = np.zeros_like(cdf)

        grid_cdf = np.linspace(0, 1, 2001)
        grid_scale = interp1d(
            cdf,
            scaled_bins,
            kind="next",
            bounds_error=False,
            fill_value=(scaled_bins[0], scaled_bins[-1]),
        )(grid_cdf)

        uniform_to_pe_arr.append(grid_scale)

    spe_scaling_factor_distributions = np.stack(uniform_to_pe_arr)
    return spe_scaling_factor_distributions


def pmt_transit_time_spread(
    _photon_timings,
    pmt_transit_time_mean,
    pmt_transit_time_spread,
    rng,
):
    """Function to add the PMT transit time spread to the photon timings.

    This function is used in the S1 and S2 photon propagation plugins.
    """
    # Note that PMT datasheet provides FWHM TTS, so sigma = TTS/(2*sqrt(2*log(2)))=TTS/2.35482
    _photon_timings += rng.normal(
        pmt_transit_time_mean, pmt_transit_time_spread / 2.35482, len(_photon_timings)
    ).astype(np.int64)

    return _photon_timings


def photon_gain_calculation(
    _photon_channels,
    p_double_pe_emision,
    gains,
    spe_scaling_factor_distributions,
    rng,
):
    """Function to calculate the PMT gain a photon will be amplified with in
    the waveform simulation."""

    # Sample if the photon is a double PE emission
    _photon_is_dpe = rng.binomial(n=1, p=p_double_pe_emision, size=len(_photon_channels)).astype(
        np.bool_
    )

    # Rename this function...
    spe_scaling_factors = sample_spe_scaling_factors(
        rng.random(len(_photon_channels)), _photon_channels, spe_scaling_factor_distributions
    )
    _photon_gains = gains[_photon_channels] * spe_scaling_factors

    # Add some double photoelectron emission by adding another sampled gain
    n_double_pe = _photon_is_dpe.sum()
    spe_scaling_factors_dpe = sample_spe_scaling_factors(
        rng.random(n_double_pe), _photon_channels[_photon_is_dpe], spe_scaling_factor_distributions
    )
    _photon_gains[_photon_is_dpe] += (
        gains[_photon_channels[_photon_is_dpe]] * spe_scaling_factors_dpe
    )

    return _photon_gains, _photon_is_dpe


def build_photon_propagation_output(
    dtype,
    _photon_timings,
    _photon_channels,
    _photon_gains,
    _photon_is_dpe,
    _cluster_id,
    photon_type,
):
    result = np.zeros(_photon_channels.shape[0], dtype=dtype)
    result["time"] = _photon_timings
    result["channel"] = _photon_channels
    result["endtime"] = result["time"]
    result["photon_gain"] = _photon_gains
    result["dpe"] = _photon_is_dpe
    result["cluster_id"] = _cluster_id
    result["photon_type"] = photon_type

    return result


def pmt_gains(to_pe, digitizer_voltage_range, digitizer_bits, pmt_circuit_load_resistor):
    """Build PMT Gains from PMT gain model and digitizer parameters."""

    adc_2_current = digitizer_voltage_range / 2 ** (digitizer_bits) / pmt_circuit_load_resistor

    gains = np.divide(
        adc_2_current,
        to_pe,
        out=np.zeros_like(to_pe),
        where=to_pe != 0,
    )
    return gains
