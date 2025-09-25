import strax
import straxen
import numpy as np
import numba

from ...dtypes import (
    primary_positions_fields,
    cluster_positions_fields,
    cluster_id_fields,
    cluster_misc_fields,
)
from ...plugin import FuseBasePlugin
from ...common import stable_sort

export, __all__ = strax.exporter()


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

    __version__ = "0.4.0"
    depends_on = ("geant4_interactions", "cluster_index")
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

    # Config options
    tag_cluster_by = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?fmt=json&take=tag_cluster_by",
        cache=True,
        help="Decide if you tag the cluster "
        "according to first interaction (time) or most energetic (energy) one",
    )

    def compute(self, geant4_interactions):

        # Remove interactions with no energy deposition
        geant4_interactions = geant4_interactions[geant4_interactions["ed"] > 0]

        if len(geant4_interactions) == 0:
            return np.zeros(0, dtype=self.dtype)

        if self.tag_cluster_by == "energy":
            cluster_by_energy = True
        elif self.tag_cluster_by == "time":
            cluster_by_energy = False
        else:
            raise ValueError("tag_cluster_by must be 'energy' or 'time'")

        # Need to sort clusters first by cluster id
        geant4_interactions_sorted = stable_sort(geant4_interactions, order="cluster_ids")
        geant4_ids = np.unique(geant4_interactions_sorted["cluster_ids"])

        result = np.zeros(len(geant4_ids), dtype=self.dtype)
        result = cluster_and_classify(result, geant4_interactions_sorted, cluster_by_energy)

        result["endtime"] = result["time"]

        return stable_sort(result, order="time")


@numba.njit
def cluster_and_classify(results, interactions, tag_cluster_by_energy):
    event_i = 0
    current_id = interactions[0]["cluster_ids"]

    weighted_x = 0
    weighted_y = 0
    weighted_z = 0
    weighted_t = 0
    weighted_time = 0
    sum_ed = 0

    largest_ed = 0
    smallest_time = 9223372036854775807  # Int64 inf
    main_interaction_index = 0

    A = 0
    Z = 0
    nestid = 0
    x_pri = 0
    y_pri = 0
    z_pri = 0
    eventid = 0
    cluster_id = 0

    for cluster_i, cluster in enumerate(interactions):

        _is_new_cluster = current_id < cluster["cluster_ids"]
        if _is_new_cluster:
            # First store results of currten cluster:
            results[event_i]["x"] = weighted_x / sum_ed
            results[event_i]["y"] = weighted_y / sum_ed
            results[event_i]["z"] = weighted_z / sum_ed
            results[event_i]["t"] = weighted_t / sum_ed
            results[event_i]["time"] = weighted_time / sum_ed
            results[event_i]["ed"] = sum_ed

            # Only call this function onces since it has to handle strings...
            A, Z, nestid = classify(
                interactions["type"][main_interaction_index],
                interactions["parenttype"][main_interaction_index],
                interactions["creaproc"][main_interaction_index],
                interactions["edproc"][main_interaction_index],
            )

            results[event_i]["A"] = A
            results[event_i]["Z"] = Z
            results[event_i]["nestid"] = nestid
            results[event_i]["x_pri"] = x_pri
            results[event_i]["y_pri"] = y_pri
            results[event_i]["z_pri"] = z_pri
            results[event_i]["eventid"] = eventid
            results[event_i]["cluster_id"] = cluster_id

            # Now prepare buffer for new cluster:
            event_i += 1
            current_id = cluster["cluster_ids"]

            # use zero here to compute average on the fly
            weighted_x = 0
            weighted_y = 0
            weighted_z = 0
            weighted_t = 0
            weighted_time = 0
            sum_ed = 0

            # Set unsued value as buffer:
            main_interaction_index = 0
            largest_ed = 0
            smallest_time = 9223372036854775807  # Int64 inf
            A = 0
            Z = 0
            nestid = 0
            x_pri = 0
            y_pri = 0
            z_pri = 0
            eventid = 0
            cluster_id = 0

        weighted_x += cluster["x"] * cluster["ed"]
        weighted_y += cluster["y"] * cluster["ed"]
        weighted_z += cluster["z"] * cluster["ed"]
        weighted_t += cluster["t"] * cluster["ed"]
        weighted_time += cluster["time"] * cluster["ed"]
        sum_ed += cluster["ed"]

        largest_ed = max(largest_ed, cluster["ed"])
        smallest_time = min(smallest_time, cluster["time"])

        if tag_cluster_by_energy and (cluster["ed"] == largest_ed):
            main_interaction_index = cluster_i
            eventid = cluster["eventid"]
            cluster_id = cluster["cluster_ids"]
            x_pri = cluster["x_pri"]
            y_pri = cluster["y_pri"]
            z_pri = cluster["z_pri"]

        elif not tag_cluster_by_energy and smallest_time == cluster["time"]:
            main_interaction_index = cluster_i
            eventid = cluster["eventid"]
            cluster_id = cluster["cluster_ids"]
            x_pri = cluster["x_pri"]
            y_pri = cluster["y_pri"]
            z_pri = cluster["z_pri"]

    # Done with looping store last result:
    results[event_i]["x"] = weighted_x / sum_ed
    results[event_i]["y"] = weighted_y / sum_ed
    results[event_i]["z"] = weighted_z / sum_ed
    results[event_i]["t"] = weighted_t / sum_ed
    results[event_i]["time"] = weighted_time / sum_ed
    results[event_i]["ed"] = sum_ed

    # Only call this function onces since it has to handle strings...
    A, Z, nestid = classify(
        interactions["type"][main_interaction_index],
        interactions["parenttype"][main_interaction_index],
        interactions["creaproc"][main_interaction_index],
        interactions["edproc"][main_interaction_index],
    )

    results[event_i]["A"] = A
    results[event_i]["Z"] = Z
    results[event_i]["nestid"] = nestid
    results[event_i]["x_pri"] = x_pri
    results[event_i]["y_pri"] = y_pri
    results[event_i]["z_pri"] = z_pri
    results[event_i]["eventid"] = eventid
    results[event_i]["cluster_id"] = cluster_id

    return results


infinity = np.iinfo(np.int8).max


@numba.njit
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
    elif (types == "e-") | (types == "e+"):
        return 0, 0, 8
    else:
        return infinity, infinity, 12
