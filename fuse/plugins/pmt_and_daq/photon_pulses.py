import numpy as np
import numba
import strax
import straxen

from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()


@export
class PulseWindow(FuseBasePlugin):
    """Plugin to compute time intervals (called ``pulse_windows``) in which the
    PMT response of photons can overlap.

    Additionally a ``pulse_id`` is computed
    for each propagated photon to identify the pulse window it belongs to.
    """

    __version__ = "0.2.1"

    depends_on = "photon_summary"

    provides = ("pulse_windows", "pulse_ids")
    data_kind = {"pulse_windows": "pulse_windows", "pulse_ids": "propagated_photons"}

    dtype_pulse_windows = strax.interval_dtype + [
        ((("ID of the pulse window", "pulse_id")), np.int64)
    ]
    dtype_pulse_ids = strax.time_fields + [
        (("Pulse id to map the photon to the pulse window", "pulse_id"), np.int64)
    ]

    dtype = dict()
    dtype["pulse_windows"] = dtype_pulse_windows
    dtype["pulse_ids"] = dtype_pulse_ids

    save_when = strax.SaveWhen.TARGET

    pulse_ids_seen = 0

    # Config options
    dt = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=sample_duration",
        type=int,
        cache=True,
        help="Width of one sample [ns]",
    )

    samples_after_pulse_center = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=samples_after_pulse_center",
        type=(int, float),
        cache=True,
        help="Number of samples after the pulse center",
    )

    samples_to_store_after = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=samples_to_store_after",
        type=(int, float),
        cache=True,
        help="Number of samples to store after the pulse center",
    )

    samples_before_pulse_center = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=samples_before_pulse_center",
        type=(int, float),
        cache=True,
        help="Number of samples before the pulse center",
    )

    samples_to_store_before = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=samples_to_store_before",
        type=(int, float),
        cache=True,
        help="Number of samples to store before the pulse center",
    )

    n_tpc_pmts = straxen.URLConfig(
        type=int,
        help="Number of PMTs in the TPC",
    )

    def setup(self):
        super().setup()

        # Lets double the samples_to_store_x values to avoid
        # overlapping records when triggering on noise..
        self.pulse_left_extenstion = (
            np.int64(2 * self.samples_to_store_before) + self.samples_before_pulse_center
        )
        self.pulse_right_extenstion = (
            np.int64(2 * self.samples_to_store_after) + self.samples_after_pulse_center
        )

    def compute(self, propagated_photons, start, end):
        if len(propagated_photons) == 0:
            return {
                "pulse_windows": np.zeros(0, self.dtype["pulse_windows"]),
                "pulse_ids": np.zeros(0, self.dtype["pulse_ids"]),
            }

        single_photon_pulses = np.zeros(len(propagated_photons), dtype=strax.interval_dtype)

        single_photon_pulses["length"] = (
            self.samples_before_pulse_center + self.samples_after_pulse_center
        )
        single_photon_pulses["dt"] = self.dt
        single_photon_pulses["time"] = propagated_photons["time"]
        single_photon_pulses["channel"] = propagated_photons["channel"]

        photon_pulses, photon_id = concat_overlapping_hits(
            single_photon_pulses,
            (self.pulse_left_extenstion, self.pulse_right_extenstion),
            (0, self.n_tpc_pmts),
            start,
            end,
        )
        photon_pulses["pulse_id"] += self.pulse_ids_seen

        pulse_ids = np.zeros(len(photon_id), self.dtype["pulse_ids"])
        pulse_ids["pulse_id"] = photon_id + self.pulse_ids_seen
        pulse_ids["time"] = propagated_photons["time"]
        pulse_ids["endtime"] = propagated_photons["endtime"]

        self.pulse_ids_seen += photon_id.max()

        return {
            "pulse_windows": strax.sort_by_time(photon_pulses),
            "pulse_ids": strax.sort_by_time(pulse_ids),
        }


# Modified code taken from strax:
# https://github.com/AxFoundation/strax/blob/2fb4d1dd7186c81e797aa2773701cf3d693a1d67/strax/processing/hitlets.py#L55C1-L156
def concat_overlapping_hits(hits, extensions, pmt_channels, start, end):
    """Function which concatenates hits which may overlap after left and right
    hit extension. Assumes that hits are sorted correctly.

    Note:
        This function only updates time, and length of the hit.

    Args:
        hits: Hits in records
        extensions: Tuple of the left and right hit extension
        pmt_channels: Tuple of the detectors first and last PMT
        start: Startime of the chunk
        end: Endtime of the chunk

    Returns:
        array with concataneted hits
    """
    first_channel, last_channel = pmt_channels
    nchannels = last_channel - first_channel + 1

    # Buffer for concat_overlapping_hits, if specified in
    # _concat_overlapping_hits numba crashes.
    last_hit_in_channel = np.zeros(
        nchannels,
        dtype=(
            strax.interval_dtype
            + [
                (("End time of the interval (ns since unix epoch)", "endtime"), np.int64),
                ((("ID of the pulse window", "pulse_id")), np.int64),
            ]
        ),
    )

    pulse_id = 0
    photon_identifiers = np.zeros(len(hits), dtype=np.int64)

    if len(hits):
        hits = _concat_overlapping_hits(
            hits,
            extensions,
            first_channel,
            last_hit_in_channel,
            photon_identifiers,
            pulse_id,
            start,
            end,
        )
    return hits, photon_identifiers


pulse_dtype = strax.interval_dtype + [((("ID of the pulse window", "pulse_id")), np.int64)]


@strax.utils.growing_result(pulse_dtype, chunk_size=int(1e4))
@numba.njit(nogil=True, cache=True)
def _concat_overlapping_hits(
    hits,
    extensions,
    first_channel,
    last_hit_in_channel_buffer,
    photon_identifiers,
    pulse_id,
    chunk_start=0,
    chunk_end=float("inf"),
    _result_buffer=None,
):
    buffer = _result_buffer
    res_offset = 0

    left_extension, right_extension = extensions
    dt = hits["dt"][0]
    assert np.all(hits["dt"] == dt), "All hits must have the same dt!"

    for i, hit in enumerate(hits):
        time_with_le = hit["time"] - int(left_extension * hit["dt"])
        endtime_with_re = strax.endtime(hit) + int(right_extension * hit["dt"])
        hit_channel = hit["channel"]

        last_hit_in_channel = last_hit_in_channel_buffer[hit_channel - first_channel]

        found_no_hit_for_channel_yet = last_hit_in_channel["time"] == 0
        if found_no_hit_for_channel_yet:
            last_hit_in_channel["time"] = max(time_with_le, chunk_start)
            last_hit_in_channel["endtime"] = min(endtime_with_re, chunk_end)
            last_hit_in_channel["channel"] = hit_channel
            last_hit_in_channel["dt"] = dt

            last_hit_in_channel["pulse_id"] = pulse_id
            photon_identifiers[i] = last_hit_in_channel["pulse_id"]
            pulse_id += 1

        else:
            hits_overlap_in_channel = last_hit_in_channel["endtime"] >= time_with_le
            if hits_overlap_in_channel:
                last_hit_in_channel["endtime"] = endtime_with_re
                photon_identifiers[i] = last_hit_in_channel["pulse_id"]
            else:
                # No, this means we have to save the previous data and update lhc:
                res = buffer[res_offset]
                res["time"] = last_hit_in_channel["time"]
                hitlet_length = last_hit_in_channel["endtime"] - last_hit_in_channel["time"]
                hitlet_length //= last_hit_in_channel["dt"]
                res["length"] = hitlet_length
                res["channel"] = last_hit_in_channel["channel"]
                res["dt"] = last_hit_in_channel["dt"]
                res["pulse_id"] = last_hit_in_channel["pulse_id"]

                # Updating current last hit:
                last_hit_in_channel["time"] = time_with_le
                last_hit_in_channel["endtime"] = endtime_with_re
                last_hit_in_channel["channel"] = hit_channel

                last_hit_in_channel["pulse_id"] = pulse_id
                photon_identifiers[i] = last_hit_in_channel["pulse_id"]
                pulse_id += 1

                res_offset += 1
                if res_offset == len(buffer):
                    yield res_offset
                    res_offset = 0

    # We went through so now we have to save all remaining hits:
    mask = last_hit_in_channel_buffer["time"] != 0
    for last_hit_in_channel in last_hit_in_channel_buffer[mask]:
        res = buffer[res_offset]
        res["time"] = last_hit_in_channel["time"]
        res["channel"] = last_hit_in_channel["channel"]
        hitlet_length = last_hit_in_channel["endtime"] - last_hit_in_channel["time"]
        hitlet_length //= last_hit_in_channel["dt"]
        res["length"] = hitlet_length
        res["dt"] = last_hit_in_channel["dt"]
        res["pulse_id"] = last_hit_in_channel["pulse_id"]

        res_offset += 1
        if res_offset == len(buffer):
            yield res_offset
            res_offset = 0
    yield res_offset
