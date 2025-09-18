import strax
import numpy as np
import numba

from ...dtypes import (
    primary_positions_fields,
    cluster_positions_fields,
    cluster_id_fields,
    cluster_misc_fields,
)
from ...common import stable_sort
from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()


@export
class MergeLineage(FuseBasePlugin):
    """Plugin that merges energy deposits with the same lineage_index into a
    single interaction.

    The 3D postiion is calculated as the energy weighted average of the
    3D positions of the energy deposits. The time of the merged lineage
    is calculated as the energy weighted average of the times of the
    energy deposits. The energy of the merged lineage is the sum of the
    individual energy depositions.
    """

    __version__ = "0.0.3"

    depends_on = ("geant4_interactions", "interaction_lineage")

    provides = "clustered_interactions"
    data_kind = "clustered_interactions"

    save_when = strax.SaveWhen.TARGET

    dtype = (
        cluster_positions_fields
        + cluster_id_fields
        + cluster_misc_fields
        + primary_positions_fields
        + strax.time_fields
    )

    def compute(self, geant4_interactions):

        # Remove all clusters that have no energy deposition
        geant4_interactions = geant4_interactions[geant4_interactions["ed"] > 0]

        if len(geant4_interactions) == 0:
            return np.zeros(0, dtype=self.dtype)

        geant4_interactions = stable_sort(geant4_interactions, order="lineage_index")
        result = np.zeros(len(np.unique(geant4_interactions["lineage_index"])), dtype=self.dtype)
        result = merge_lineages(result, geant4_interactions)

        result["endtime"] = result["time"]

        return stable_sort(result, order="time")


@numba.njit
def merge_lineages(results, interactions):
    event_i = 0
    current_lineage = interactions[0]["lineage_index"]

    weighted_x = 0
    weighted_y = 0
    weighted_z = 0
    weighted_t = 0
    weighted_time = 0
    sum_ed = 0

    lineage_index = -1
    eventid = -1
    track_id = -1
    nestid = -1
    A = -1
    Z = -1
    x_pri = -1
    y_pri = -1
    z_pri = -1

    for lineage_i, lineage in enumerate(interactions):

        _is_new_lineage = current_lineage < lineage["lineage_index"]
        if _is_new_lineage:
            # First store results of currten cluster:
            results[event_i]["x"] = weighted_x / sum_ed
            results[event_i]["y"] = weighted_y / sum_ed
            results[event_i]["z"] = weighted_z / sum_ed
            results[event_i]["t"] = weighted_t / sum_ed
            results[event_i]["time"] = weighted_time / sum_ed
            results[event_i]["ed"] = sum_ed

            results[event_i]["cluster_id"] = lineage_index

            results[event_i]["eventid"] = eventid
            results[event_i]["trackid"] = track_id
            results[event_i]["nestid"] = nestid
            results[event_i]["A"] = A
            results[event_i]["Z"] = Z
            results[event_i]["x_pri"] = x_pri
            results[event_i]["y_pri"] = y_pri
            results[event_i]["z_pri"] = z_pri

            # Now prepare buffer for new cluster:
            event_i += 1
            current_lineage = lineage["lineage_index"]

            # use zero here to compute average on the fly
            weighted_x = 0
            weighted_y = 0
            weighted_z = 0
            weighted_t = 0
            weighted_time = 0
            sum_ed = 0

            lineage_index = -1
            eventid = -1
            track_id = -1
            nestid = -1
            A = -1
            Z = -1
            x_pri = -1
            y_pri = -1
            z_pri = -1

        weighted_x += lineage["x"] * lineage["ed"]
        weighted_y += lineage["y"] * lineage["ed"]
        weighted_z += lineage["z"] * lineage["ed"]
        weighted_t += lineage["t"] * lineage["ed"]
        weighted_time += lineage["time"] * lineage["ed"]
        sum_ed += lineage["ed"]

        _is_first_lineage = lineage_index == -1
        if _is_first_lineage:
            lineage_index = lineage["lineage_index"]
            eventid = lineage["eventid"]
            track_id = lineage["lineage_trackid"]
            nestid = lineage["lineage_type"]
            A = lineage["A"]
            Z = lineage["Z"]
            x_pri = lineage["x_pri"]
            y_pri = lineage["y_pri"]
            z_pri = lineage["z_pri"]

    # First store results of currten cluster:
    results[event_i]["x"] = weighted_x / sum_ed
    results[event_i]["y"] = weighted_y / sum_ed
    results[event_i]["z"] = weighted_z / sum_ed
    results[event_i]["t"] = weighted_t / sum_ed
    results[event_i]["time"] = weighted_time / sum_ed
    results[event_i]["ed"] = sum_ed

    results[event_i]["cluster_id"] = lineage_index

    results[event_i]["eventid"] = eventid
    results[event_i]["trackid"] = track_id
    results[event_i]["nestid"] = nestid
    results[event_i]["A"] = A
    results[event_i]["Z"] = Z
    results[event_i]["x_pri"] = x_pri
    results[event_i]["y_pri"] = y_pri
    results[event_i]["z_pri"] = z_pri

    return results
