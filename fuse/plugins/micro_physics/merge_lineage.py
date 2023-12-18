import strax
import straxen
import numpy as np
import numba
import logging

export, __all__ = strax.exporter()

from ...common import FUSE_PLUGIN_TIMEOUT

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.micro_physics.merge_lineage')

@export
class MergeLineage(strax.Plugin):
    
    __version__ = "0.0.1"
    
    depends_on = ("geant4_interactions", "interaction_lineage")
    
    provides = "clustered_interactions"
    data_kind = "clustered_interactions"

    #Forbid rechunking
    rechunk_on_save = False

    save_when = strax.SaveWhen.TARGET

    input_timeout = FUSE_PLUGIN_TIMEOUT
    
    dtype = [('x', np.float32),
             ('y', np.float32),
             ('z', np.float32),
             ('ed', np.float32),
             ('nestid', np.int8),
             ('A', np.int16),
             ('Z', np.int16),
             ('evtid', np.int32),
             ('xe_density', np.float32), #Will be set i a later plugin
             ('vol_id', np.int8), #Will be set i a later plugin
             ('create_S2', np.bool8), #Will be set i a later plugin
            ]
    
    dtype = dtype + strax.time_fields

    #Config options
    debug = straxen.URLConfig(
        default=False, type=bool,track=False,
        help='Show debug informations',
    )
    
    def setup(self):

        if self.debug:
            log.setLevel('DEBUG')
            log.debug(f"Running MergeLineage version {self.__version__} in debug mode")
        else: 
            log.setLevel('INFO')

    def compute(self, geant4_interactions):

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

        #These ones are the same for all interactions in the lineage
        result[i]["evtid"] = lineage["evtid"][0] 
        result[i]["nestid"] = lineage["lineage_type"][0]
        result[i]["A"] = lineage["A"][0]
        result[i]["Z"] = lineage["Z"][0]

    return result