import strax
import straxen
import numpy as np
import logging

from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.micro_physics.merge_cluster")


@export
class MergeCluster(FuseBasePlugin):
    """Plugin that merges energy deposits with the same cluster index into a
    single interaction.

    The 3D postiion is calculated as the energy weighted average of the
    3D positions of the energy deposits. The time of the merged cluster
    is calculated as the energy weighted average of the times of the
    energy deposits. The energy of the merged cluster is the sum of the
    individual energy depositions. The cluster is then classified based
    on either the first interaction in the cluster or the most energetic
    interaction.
    """

    __version__ = "0.3.0"

    depends_on = ("geant4_interactions", "cluster_index")

    provides = "clustered_interactions"
    data_kind = "clustered_interactions"

    save_when = strax.SaveWhen.TARGET

    dtype = [
        (("x position of the cluster [cm]", "x"), np.float32),
        (("y position of the cluster [cm]", "y"), np.float32),
        (("z position of the cluster [cm]", "z"), np.float32),
        (("Energy of the cluster [keV]", "ed"), np.float32),
        (("NEST interaction type", "nestid"), np.int8),
        (("Mass number of the interacting particle", "A"), np.int8),
        (("Charge number of the interacting particle", "Z"), np.int8),
        (("Geant4 event ID", "evtid"), np.int32),
        (("x position of the primary particle [cm]", "x_pri"), np.float32),
        (("y position of the primary particle [cm]", "y_pri"), np.float32),
        (("z position of the primary particle [cm]", "z_pri"), np.float32),
        (("ID of the cluster", "cluster_id"), np.int32),
        (("Xenon density at the cluster position. Will be set later.", "xe_density"), np.float32),
        (("ID of the volume in which the cluster occured. Will be set later.", "vol_id"), np.int8),
        (
            (
                "Flag indicating if a cluster can create a S2 signal. Will be set later.",
                "create_S2",
            ),
            np.bool_,
        ),
    ]

    dtype = dtype + strax.time_fields

    # Config options
    tag_cluster_by = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?fmt=json&take=tag_cluster_by",
        cache=True,
        help="Decide if you tag the cluster "
        "according to first interaction (time) or most energetic (energy) one.",
    )

    def compute(self, geant4_interactions):
        if len(geant4_interactions) == 0:
            return np.zeros(0, dtype=self.dtype)

        result = np.zeros(len(np.unique(geant4_interactions["cluster_ids"])), dtype=self.dtype)
        result = cluster_and_classify(result, geant4_interactions, self.tag_cluster_by)

        result["endtime"] = result["time"]

        return result


# @numba.njit()
def cluster_and_classify(result, interactions, tag_cluster_by):
    interaction_cluster = [
        interactions[interactions["cluster_ids"] == i]
        for i in np.unique(interactions["cluster_ids"])
    ]

    for i, cluster in enumerate(interaction_cluster):
        result[i]["x"] = np.average(cluster["x"], weights=cluster["ed"])
        result[i]["y"] = np.average(cluster["y"], weights=cluster["ed"])
        result[i]["z"] = np.average(cluster["z"], weights=cluster["ed"])
        result[i]["time"] = np.average(cluster["time"], weights=cluster["ed"])
        result[i]["ed"] = np.sum(cluster["ed"])

        if tag_cluster_by == "energy":
            main_interaction_index = np.argmax(cluster["ed"])
        elif tag_cluster_by == "time":
            main_interaction_index = np.argmin(cluster["time"])
        else:
            raise ValueError("tag_cluster_by must be 'energy' or 'time'")

        A, Z, nestid = classify(
            cluster["type"][main_interaction_index],
            cluster["parenttype"][main_interaction_index],
            cluster["creaproc"][main_interaction_index],
            cluster["edproc"][main_interaction_index],
        )
        result[i]["A"] = A
        result[i]["Z"] = Z
        result[i]["nestid"] = nestid

        result[i]["x_pri"] = cluster["x_pri"][main_interaction_index]
        result[i]["y_pri"] = cluster["y_pri"][main_interaction_index]
        result[i]["z_pri"] = cluster["z_pri"][main_interaction_index]
        result[i]["evtid"] = cluster["evtid"][main_interaction_index]

        # Get cluster id from and save it!
        result[i]["cluster_id"] = cluster["cluster_ids"][main_interaction_index]

    return result


infinity = np.iinfo(np.int8).max


def classify(types, parenttype, creaproc, edproc):
    "Function to classify a cluster according to its main interaction"

    if (edproc == "ionIoni") & (types != "alpha"):
        return 0, 0, 0
    elif (types == "neutron") & (edproc == "hadElastic"):
        return 0, 0, 0
    elif types == "alpha":
        return 4, 2, 6
    elif parenttype == "Kr83[9.405]":
        return infinity, 0, 11
    elif parenttype == "Kr83[41.557]":
        return infinity, 0, 11
    elif types == "gamma":
        return 0, 0, 7
    elif types == "e-":
        return 0, 0, 8
    else:
        return infinity, infinity, 12
