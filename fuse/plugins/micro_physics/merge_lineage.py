import strax
import numpy as np

from ...dtypes import (
    primary_positions_fields,
    cluster_positions_fields,
    cluster_id_fields,
    cluster_misc_fields,
)
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

    __version__ = "0.0.1"

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

        result = np.zeros(len(np.unique(geant4_interactions["lineage_index"])), dtype=self.dtype)
        result = merge_lineages(result, geant4_interactions)

        result["endtime"] = result["time"]

        return result


def merge_lineages(result, interactions):

    lineages_in_event = [
        interactions[interactions["lineage_index"] == i]
        for i in np.unique(interactions["lineage_index"])
    ]

    for i, lineage in enumerate(lineages_in_event):

        result[i]["x"] = np.average(lineage["x"], weights=lineage["ed"])
        result[i]["y"] = np.average(lineage["y"], weights=lineage["ed"])
        result[i]["z"] = np.average(lineage["z"], weights=lineage["ed"])
        result[i]["time"] = np.average(lineage["time"], weights=lineage["ed"])
        result[i]["ed"] = np.sum(lineage["ed"])

        result[i]["cluster_id"] = lineage["lineage_index"][0]

        # These ones are the same for all interactions in the lineage
        result[i]["eventid"] = lineage["eventid"][0]
        result[i]["nestid"] = lineage["lineage_type"][0]
        result[i]["A"] = lineage["A"][0]
        result[i]["Z"] = lineage["Z"][0]
        result[i]["x_pri"] = lineage["x_pri"][0]
        result[i]["y_pri"] = lineage["y_pri"][0]
        result[i]["z_pri"] = lineage["z_pri"][0]

    return result
