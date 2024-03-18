import strax
import numpy as np
import logging

export, __all__ = strax.exporter()

from ...plugin import FuseBasePlugin

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.micro_physics.merge_lineage')

@export
class MergeLineage(strax.Plugin):
    
    __version__ = "0.0.1"
    
    depends_on = ("geant4_interactions", "interaction_lineage")
    
    provides = "clustered_interactions"
    data_kind = "clustered_interactions"

    save_when = strax.SaveWhen.TARGET
    
    dtype = [(("x position of the cluster [cm]", "x"), np.float32),
             (("y position of the cluster [cm]", "y"), np.float32),
             (("z position of the cluster [cm]", "z"), np.float32),
             (("Energy of the cluster [keV]", "ed"), np.float32),
             (("NEST interaction type", "nestid"), np.int8),
             (("Mass number of the interacting particle", "A"), np.int16),
             (("Charge number of the interacting particle", "Z"), np.int16),
             (("Geant4 event ID", "evtid"), np.int32),
             #(("x position of the primary particle [cm]", "x_pri"), np.float32),
             #(("y position of the primary particle [cm]", "y_pri"), np.float32),
             #(("z position of the primary particle [cm]", "z_pri"), np.float32),
             (("ID of the cluster", "cluster_id"), np.int32),
             (("Xenon density at the cluster position.", "xe_density"), np.float32), 
             (("ID of the volume in which the cluster occured.", "vol_id"), np.int8),
             (("Flag indicating if a cluster can create a S2 signal.", "create_S2"), np.bool_),
            # (("Type of the main cluster (alpha beta gamma)", "main_cluster_type"), np.dtype("U5")),
            ]
    
    dtype = dtype + strax.time_fields

    def compute(self, geant4_interactions):

        #Remove all clusters that have no energy deposition
        geant4_interactions = geant4_interactions[geant4_interactions["ed"] > 0]

        if len(geant4_interactions) == 0:
            return np.zeros(0, dtype=self.dtype)

        result = np.zeros(len(np.unique(geant4_interactions["lineage_index"])), dtype=self.dtype)
        result = merge_lineages(result, geant4_interactions)

        result["endtime"] = result["time"]
        
        return result

def merge_lineages(result, interactions):

    lineages_in_event = [interactions[interactions["lineage_index"] == i] for i in np.unique(interactions["lineage_index"])]
    
    for i, lineage in enumerate(lineages_in_event):

        result[i]["x"] = np.average(lineage["x"], weights = lineage["ed"])
        result[i]["y"] = np.average(lineage["y"], weights = lineage["ed"])
        result[i]["z"] = np.average(lineage["z"], weights = lineage["ed"])
        result[i]["time"] = np.average(lineage["time"], weights = lineage["ed"])
        result[i]["ed"] = np.sum(lineage["ed"])

        
        result[i]["cluster_id"] = lineage["lineage_index"][0]

        #These ones are the same for all interactions in the lineage
        result[i]["evtid"] = lineage["evtid"][0] 
        result[i]["nestid"] = lineage["lineage_type"][0]
        result[i]["A"] = lineage["A"][0]
        result[i]["Z"] = lineage["Z"][0]
        # result[i]["main_cluster_type"] = lineage["main_cluster_type"][0]

    return result