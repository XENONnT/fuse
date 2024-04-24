import logging

import numpy as np
from numba import njit
from numba.typed import List
from scipy.interpolate import interp1d
import strax
import straxen

from ...plugin import FuseBaseDownChunkingPlugin
from .pmt_response_and_daq import PMTResponseAndDAQ, find_split_index, split_photons
from .photon_pulses import concat_overlapping_hits
from ..micro_physics.find_cluster import simple_1d_clustering

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.pmt_and_daq.sprinkled_records")


@export
class SprinkledRecords(PMTResponseAndDAQ):
    """Plugin to simulate sprinkled raw records.

    In addition to what the PMTResponseAndDAQ simulates, it mixes the
    simulated raw records with the ones from real data.
    """

    __version__ = "0.0.1"

    child_plugin = True

    save_when = strax.SaveWhen.TARGET

    # Config options
    raw_records_st_module = straxen.URLConfig(
        type=str,
        help="The module where the context for raw_records loading is defined."
        'Can be "straxen", "cutax", or "fuse".',
    )

    raw_records_st_name = straxen.URLConfig(
        type=str,
        help="The name of the context where raw_records is stored and can be loaded."
        'Can be "straxen", "cutax", or "fuse".',
    )

    raw_records_st_kwargs = straxen.URLConfig(
        type=dict, help="Keyword arguments for context initialization"
    )

    raw_records_st_config = straxen.URLConfig(
        type=dict, help="Configuration of context for raw_records"
    )

    sprinkle_run_id = straxen.URLConfig(
        type=str, help="id of a run whose raw_records will be used for sprinkling"
    )

    tpc_length = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?&fmt=json&take=tpc_length",
        type=(int, float),
        cache=True,
        help="Length of the XENONnT TPC [cm]",
    )

    drift_velocity_liquid = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=drift_velocity_liquid",
        type=(int, float),
        cache=True,
        help="Drift velocity of electrons in the liquid xenon [cm/ns]",
    )

    peaklet_gap_threshold = straxen.URLConfig(
        default=700, infer_type=False, help="No hits for this many ns triggers a new peak"
    )

    def setup(self):
        super().setup()
        if self.raw_records_st_module == "straxen":
            self.raw_records_st = eval(
                f"straxen.contexts.{self.raw_records_st_name}(**{self.raw_records_st_kwargs})",
                {"straxen": straxen},
            )
        elif self.raw_records_st_module == "cutax":
            import cutax

            self.raw_records_st = eval(
                f"cutax.contexts.{self.raw_records_st_name}(**{self.raw_records_st_kwargs})",
                {"cutax": cutax},
            )
        elif self.raw_records_st_module == "fuse":
            # Rare case where the raw_records is simulated. Designed for testing,
            # but maybe it is useful.
            from ...context import full_chain_context

            if self.raw_records_st_name == "full_chain_context":
                self.raw_records_st = full_chain_context(**(self.raw_records_st_kwargs))
            else:
                raise ValueError(
                    "Only full_chain_context is supported if using fuse context for sprinkling!"
                )
        else:
            raise ValueError(
                f"Unknown raw_records_st_module {self.raw_records_st_module}! "
                'Recognized modules are "straxen", "cutax" or "fuse".'
            )
        self.raw_records_st_sanity_check()
        self.raw_records_st.set_config(self.raw_records_st_config)

        # Get time/endtime of a run
        sprinkle_run_info = self.raw_records_st.select_runs(run_id=self.sprinkle_run_id)
        if len(sprinkle_run_info == 1):
            self.sprinkle_run_start = sprinkle_run_info["time"].iloc(0)
            self.sprinkle_run_end = sprinkle_run_info["end"].iloc(0)
        else:
            log.warning(
                "Can't get run metadata. Assuming that this is an MC run. "
                "Try to load raw_records to memory. THIS MAY CAUSE OOM!"
            )
            raw_records_data = self.raw_records_st.get_array(self.sprinkle_run_id, "raw_records")
            self.sprinkle_run_start = np.min(raw_records_data["time"])
            self.sprinkle_run_end = np.max(
                raw_records_data["time"] + raw_records_data["length"] * raw_records_data["dt"]
            )

    def compute(self, propagated_photons, pulse_windows, start, end):
        if len(propagated_photons) == 0 or len(pulse_windows) == 0:
            log.debug("No photons or pulse windows found for chunk!")

            yield self.chunk(start=start, end=end, data=np.zeros(0, dtype=self.dtype))

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
            log.info(f"Chunk size exceeding file size target. Downchunking to {n_chunks} chunks")

        last_start = start

        photons, unique_photon_pulse_ids = split_photons(propagated_photons)

        # convert photons to numba list for njit
        _photons = List()
        [_photons.append(x) for x in photons]

        if n_chunks > 1:
            for pulse_group in pulse_window_chunks[:-1]:
                records = self.compute_chunk(_photons, unique_photon_pulse_ids, pulse_group)
                chunk_end = np.max(strax.endtime(records))
                chunk = self.chunk(start=last_start, end=chunk_end, data=records)
                last_start = chunk_end
                yield chunk

        # And the last chunk
        records = self.compute_chunk(_photons, unique_photon_pulse_ids, pulse_window_chunks[-1])
        chunk = self.chunk(start=last_start, end=end, data=records)
        yield chunk

    def compute_chunk(self, _photons, unique_photon_pulse_ids, pulse_group):
        simulated_records = super().compute_chunk(_photons, unique_photon_pulse_ids, pulse_group)
        # With pulse_group, define where to sprinkle raw_records from data
        # The time window is defined as within the 2*full-drift-time window left/right, no other
        # hits can be found.
        full_drift_time = self.tpc_length / self.drift_velocity_liquid
        # Reuse concat_overlapping_hits. Define a "fake" pulse_group
        fake_pulse_group = np.copy(pulse_group)
        fake_pulse_group["channel"] = 1
        sprinkle_time_windows, sprinkle_time_window_ids = concat_overlapping_hits(
            fake_pulse_group, (2 * full_drift_time, 2 * full_drift_time), (1, 1), 0, float("inf")
        )

        sprinkled_records = []
        # Get raw_records to sprinkle
        for sprinkle_time_window_id in sprinkle_time_window_ids:
            sprinkle_time_window_start = sprinkle_time_windows[sprinkle_time_window_id]["time"]
            sprinkle_time_window_end = (
                sprinkle_time_window_start
                + sprinkle_time_windows[sprinkle_time_window_id]["length"]
            )
            raw_records_to_sprinkle = self.get_sprinkle_raw_records(
                self.sprinkle_run_start,
                self.sprinkle_run_end,
                sprinkle_time_window_start,
                sprinkle_time_window_end,
            )
            sprinkled_records.append(raw_records_to_sprinkle)
        sprinkled_records = np.concatenate(sprinkled_records)
        records = np.concatenate((simulated_records, sprinkled_records))
        return strax.sort_by_time(records)

    def raw_records_st_sanity_check(self):
        if not isinstance(self.raw_records_st, strax.Context):
            raise ValueError(
                f"{self.raw_records_st_name} in module {self.raw_records_st_module} "
                "is not a valid context!"
            )
        if not self.raw_records_st.is_stored(self.sprinkle_run_id, "raw_records"):
            raise ValueError(f"The raw_records for {self.sprinkle_run_id} is not stored!")

    def get_sprinkle_raw_records(
        self,
        sprinkle_run_start,
        sprinkle_run_end,
        sprinkle_time_window_start,
        sprinkle_time_window_end,
    ):
        # Sample time range where the waveform from data is sprinkled
        full_drift_time = self.tpc_length / self.drift_velocity_liquid
        sprinkle_time_window_length = sprinkle_time_window_end - sprinkle_time_window_start
        assert (
            sprinkle_run_start + 4 * full_drift_time + sprinkle_time_window_length
            < sprinkle_run_end
        ), "The raw_records to sprinkle is too short."
        sprinkle_waveform_time_start = self.rng.integers(
            sprinkle_run_start + 2 * full_drift_time,
            sprinkle_run_end - 2 * full_drift_time - sprinkle_time_window_length,
            dtype=np.int64,
        )

        # Extend the time range so that st.get_array (~100ms) is only called once
        try:
            sprinkle_raw_records_candidate = self.raw_records_st.get_array(
                self.sprinkle_run_id,
                "raw_records",
                time_range=[
                    sprinkle_waveform_time_start - 2 * full_drift_time,
                    sprinkle_waveform_time_start
                    + sprinkle_time_window_length
                    + 2 * full_drift_time,
                ],
                time_selection="touching",  # Avoid cutting complete waveforms
            )
        except ValueError:
            # Expecting ValueError if st.get_array returns nothing
            return np.array([], dtype=self.dtype)
        raw_records_to_sprinkle = _get_sprinkle_raw_records(
            sprinkle_raw_records_candidate,
            sprinkle_waveform_time_start,
            sprinkle_time_window_length,
            full_drift_time,
            self.peaklet_gap_threshold,
        )
        raw_records_to_sprinkle["channel"] += self.n_tpc_pmts
        raw_records_to_sprinkle["time"] -= sprinkle_waveform_time_start
        raw_records_to_sprinkle["time"] += sprinkle_time_window_start
        return raw_records_to_sprinkle


@njit(cache=True)
def _get_sprinkle_raw_records(
    sprinkle_raw_records_candidate,
    sprinkle_waveform_time_start,
    sprinkle_time_window_length,
    full_drift_time,
    peaklet_gap_threshold,
):
    extend_before, extend_after = 0, 0
    start_time = sprinkle_raw_records_candidate["time"]
    end_time = (
        sprinkle_raw_records_candidate["time"]
        + sprinkle_raw_records_candidate["length"] * sprinkle_raw_records_candidate["dt"]
    )
    current_selection_mask = end_time >= sprinkle_waveform_time_start
    current_selection_mask &= end_time >= sprinkle_waveform_time_start

    # Increment extend_before until there is no further raw_records included
    while True:
        if peaklet_gap_threshold * (extend_before + 1) > 2 * full_drift_time:
            # log.warning("Very dirty data observed within two FDT. Stopping iteration. "
            #             "This should NOT happen.")
            break
        proposed_selection_mask = (
            end_time >= sprinkle_waveform_time_start - peaklet_gap_threshold * (extend_before + 1)
        )
        proposed_selection_mask &= (
            start_time <= sprinkle_waveform_time_start + sprinkle_time_window_length
        )
        if np.all(proposed_selection_mask == current_selection_mask):
            break
        current_selection_mask = proposed_selection_mask
        extend_before += 1

    # Repeat for extend_after
    while True:
        if peaklet_gap_threshold * (extend_after + 1) > 2 * full_drift_time:
            # log.warning("Very dirty data observed within two FDT. Stopping iteration. "
            #             "This should NOT happen.")
            break
        proposed_selection_mask = (
            end_time >= sprinkle_waveform_time_start - peaklet_gap_threshold * extend_before
        )
        proposed_selection_mask &= (
            start_time
            <= sprinkle_waveform_time_start
            + sprinkle_time_window_length
            + peaklet_gap_threshold * (extend_after + 1)
        )
        if np.all(proposed_selection_mask == current_selection_mask):
            break
        current_selection_mask = proposed_selection_mask
        extend_before += 1

    return sprinkle_raw_records_candidate[current_selection_mask]
