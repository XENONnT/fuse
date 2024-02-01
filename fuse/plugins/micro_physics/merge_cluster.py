import strax
import straxen
import numpy as np
import numba
import logging

export, __all__ = strax.exporter()

from ...plugin import FuseBasePlugin

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.micro_physics.merge_cluster')

@export
class MergeCluster(FuseBasePlugin):
    
    __version__ = "0.1.1"
    
    depends_on = ("geant4_interactions", "cluster_index")
    
    provides = "clustered_interactions"
    data_kind = "clustered_interactions"

    save_when = strax.SaveWhen.TARGET
    
    dtype = [('x', np.float32),
             ('y', np.float32),
             ('z', np.float32),
             ('ed', np.float32),
             ('nestid', np.int8),
             ('A', np.int8),
             ('Z', np.int8),
             ('evtid', np.int32),
             ('x_pri', np.float32),
             ('y_pri', np.float32),
             ('z_pri', np.float32),
             ('xe_density', np.float32), #Will be set i a later plugin
             ('vol_id', np.int8), #Will be set i a later plugin
             ('create_S2', np.bool_), #Will be set i a later plugin
            ]
    
    dtype = dtype + strax.time_fields

    #Config options
    tag_cluster_by = straxen.URLConfig(
        default="energy",
        help='decide if you tag the cluster (particle type, energy depositing process)\
              according to first interaction in it (time) or most energetic (energy))',
    )
        
    def compute(self, geant4_interactions):

        if len(geant4_interactions) == 0:
            return np.zeros(0, dtype=self.dtype)

        result = np.zeros(len(np.unique(geant4_interactions["cluster_ids"])), dtype=self.dtype)
        result = cluster_and_classify(result, geant4_interactions, self.tag_cluster_by)

        result["endtime"] = result["time"]
        
        return result

#@numba.njit()
def cluster_and_classify(result, interactions, tag_cluster_by):

    interaction_cluster = [interactions[interactions["cluster_ids"] == i] for i in np.unique(interactions["cluster_ids"])]

    for i, cluster in enumerate(interaction_cluster):
        result[i]["x"] = np.average(cluster["x"], weights = cluster["ed"])
        result[i]["y"] = np.average(cluster["y"], weights = cluster["ed"])
        result[i]["z"] = np.average(cluster["z"], weights = cluster["ed"])
        result[i]["time"] = np.average(cluster["time"], weights = cluster["ed"])
        result[i]["ed"] = np.sum(cluster["ed"])
        

        if tag_cluster_by == "energy":
            main_interaction_index = np.argmax(cluster["ed"])
        elif tag_cluster_by == "time":
            main_interaction_index = np.argmin(cluster["time"])
        else:
            raise ValueError("tag_cluster_by must be 'energy' or 'time'")

        A, Z, nestid = classify(cluster["type"][main_interaction_index],
                                cluster["parenttype"][main_interaction_index],
                                cluster["creaproc"][main_interaction_index],
                                cluster["edproc"][main_interaction_index]
                                )
        result[i]["A"] = A
        result[i]["Z"] = Z
        result[i]["nestid"] = nestid

        result[i]["x_pri"] = cluster["x_pri"][main_interaction_index]
        result[i]["y_pri"] = cluster["y_pri"][main_interaction_index]
        result[i]["z_pri"] = cluster["z_pri"][main_interaction_index]
        result[i]["evtid"] = cluster["evtid"][main_interaction_index]

    return result

infinity = np.iinfo(np.int8).max

def classify(types, parenttype, creaproc, edproc):
    "Function to classify a cluster according to its main interaction"

    if  (edproc == "ionIoni") & (types != "alpha"):
        return 0, 0, 0
    elif (types == "neutron") & (edproc == "hadElastic"):
        return 0, 0, 0
    elif (types == "alpha"):
        return 4, 2, 6
    elif (parenttype == "Kr83[9.405]"):
        return infinity, 0, 11
    elif (parenttype == "Kr83[41.557]"):
        return infinity, 0, 11
    elif (types == "gamma"):
        return 0, 0, 7
    elif (types == "e-"):
        return 0, 0, 8
    else: 
        return infinity, infinity, 12
