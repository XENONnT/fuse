import numpy as np
import numba

import strax
import straxen
from fuse.plugin import FuseBasePlugin

export, __all__ = strax.exporter()


@export
class MicroPhysicsSummary(strax.MergeOnlyPlugin):
    """MergeOnlyPlugin that summarizes the fuse microphysics simulation results
    into a single output."""

    depends_on = (
        "interactions_in_roi",
        "quanta",
        "electric_field_values",
    )
    rechunk_on_save = False
    save_when = strax.SaveWhen.ALWAYS
    provides = "microphysics_summary"
    __version__ = "0.1.0"


@export
class MicroPhysicsSummaryNRFilter(FuseBasePlugin):
    """Plugin which filters the microphysics summary for valid NR events."""

    depends_on = (
        "interactions_in_roi",
        "quanta",
        "electric_field_values",
    )
    save_when = strax.SaveWhen.ALWAYS

    rechunk_on_save = True
    chunk_target_size_mb = 0.5

    provides = "microphysics_summary"
    __version__ = "0.0.2"

    g1_photon_yield = straxen.URLConfig(
        default=0.1,
        type=(int, float),
        help="Scaled g1 x 0.8 to acocunt for corrections [pe/ph]",
    )
    g2_electron_yield = straxen.URLConfig(
        default=13.4,
        type=(int, float),
        help="Scaled g2 x 0.8 to acocunt for corrections [pe/e]",
    )

    max_s1_area = straxen.URLConfig(
        default=700,
        type=(int, float),
        help="Max S1 area [pe] for NR roi",
    )
    max_s2_area = straxen.URLConfig(
        default=3 * 10**4,
        type=(int, float),
        help="Max S2 area [pe] for NR roi",
    )

    def infer_dtype(self):
        dtypes = [self.deps[d].dtype_for(d) for d in sorted(self.depends_on)]
        return strax.merged_dtype(dtypes)

    def compute(self, interactions_in_roi):

        vertex_to_keep = filter_events(
            interactions_in_roi,
            self.g1_photon_yield,
            self.g2_electron_yield,
            self.max_s1_area,
            self.max_s2_area,
        )

        microphysics_summary = np.zeros(len(interactions_in_roi), dtype=self.dtype)
        strax.copy_to_buffer(interactions_in_roi, microphysics_summary, "_copy_microphysics")
        microphysics_summary = microphysics_summary[vertex_to_keep == 1]
        return microphysics_summary


@numba.njit
def filter_events(mps, g1, g2, max_s1, max_s2):
    """Small function to filter microphysics for valide NR events.

    We cut all events which are overshadowed by other events outside of
    our ROI excluding delayed deexcitations. To account for missing
    detector corrections one should scale g1/g2 down.
    """

    event_ids = np.unique(mps["eventid"])

    vertex_to_keep = np.ones(len(mps))

    vertex_i = 0

    for event_i in event_ids:
        start_index = vertex_i
        max_photons = 0
        max_electrons = 0
        prompt_photons = 0
        number_of_nr_interactions = 0
        start_time = mps["time"][vertex_i]

        for vertex_i in range(start_index, len(mps)):
            vertex = mps[vertex_i]

            _is_a_new_event = event_i < vertex["eventid"]
            if _is_a_new_event:
                # Next event starts break for loop and check next event
                break

            # Is prompt vertex:
            _is_prompt = (vertex["time"] - start_time) < 200  # ns
            if _is_prompt:
                prompt_photons += vertex["photons"]

            # Ignore and drop vertex if too much delayed within event:
            _vertex_is_delayed = (vertex["time"] - start_time) > 3_000_000  # ms
            if _vertex_is_delayed:
                vertex_to_keep[vertex_i] = 0
                continue

            max_photons = max(max_photons, vertex["photons"])
            max_electrons = max(max_electrons, vertex["electrons"])
            _is_nr = vertex["nestid"] == 0
            if _is_nr:
                number_of_nr_interactions += 1

        # Check if the largest interaction is still within ROI:
        _is_in_nr_roi = (
            (max_photons * g1 < max_s1)
            & (max_electrons * g2 < max_s2)
            & (number_of_nr_interactions > 0)
            & (prompt_photons * g1 < max_s1)
        )

        if not _is_in_nr_roi:
            vertex_to_keep[start_index:vertex_i] = 0
    return vertex_to_keep
