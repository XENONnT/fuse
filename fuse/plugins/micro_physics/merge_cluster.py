import strax
import straxen
import numpy as np
import numba
import logging

export, __all__ = strax.exporter()

from ...common import FUSE_PLUGIN_TIMEOUT

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.micro_physics.merge_cluster')

@export
class MergeCluster(strax.Plugin):
    
    __version__ = "0.1.0"
    
    depends_on = ("geant4_interactions", "cluster_index")
    
    provides = "clustered_interactions"
    data_kind = "clustered_interactions"

    #Forbid rechunking
    rechunk_on_save = False

    save_when = strax.SaveWhen.TARGET

    input_timeout = FUSE_PLUGIN_TIMEOUT
    
    dtype = [('x', np.float32),
             ('y', np.float32),
             ('z', np.float32),
             ('ed', np.float64),
             ('nestid', np.int64),
             ('A', np.int64),
             ('Z', np.int64),
             ('evtid', np.int64),
             ('x_pri', np.float32),
             ('y_pri', np.float32),
             ('z_pri', np.float32),
             ('xe_density', np.float32), #Will be set i a later plugin
             ('vol_id', np.int64), #Will be set i a later plugin
             ('create_S2', np.bool8), #Will be set i a later plugin
            ]
    
    dtype = dtype + strax.time_fields

    #Config options
    debug = straxen.URLConfig(
        default=False, type=bool,track=False,
        help='Show debug informations',
    )

    tag_cluster_by = straxen.URLConfig(
        default="energy",
        help='decide if you tag the cluster (particle type, energy depositing process)\
              according to first interaction in it (time) or most energetic (energy))',
    )
    
    def setup(self):

        if self.debug:
            log.setLevel('DEBUG')
            log.debug(f"Running MergeCluster version {self.__version__} in debug mode")
        else: 
            log.setLevel('WARNING')

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



infinity = np.iinfo(np.int16).max
classifier_dtype = [(('Interaction type', 'types'), np.dtype('<U30')),
                    (('Interaction type of the parent', 'parenttype'), np.dtype('<U30')),
                    (('Creation process', 'creaproc'), np.dtype('<U30')),
                    (('Energy deposit process', 'edproc'), np.dtype('<U30')),
                    (('Atomic mass number', 'A'), np.int16),
                    (('Atomic number', 'Z'), np.int16),
                    (('Nest Id for qunata generation', 'nestid'), np.int16)]
classifier = np.zeros(7, dtype=classifier_dtype)
classifier['types'] = ['None', 'neutron', 'alpha', 'None', 'None', 'gamma', 'e-']
classifier['parenttype'] = ['None', 'None', 'None', 'Kr83[9.405]', 'Kr83[41.557]', 'None', 'None']
classifier['creaproc'] = ['None', 'None', 'None', 'None', 'None', 'None', 'None']
classifier['edproc'] = ['ionIoni', 'hadElastic', 'None', 'None', 'None', 'None', 'None']
classifier['A'] = [0, 0, 4, infinity, infinity, 0, 0]
classifier['Z'] = [0, 0, 2, 0, 0, 0, 0]
classifier['nestid'] = [0, 0, 6, 11, 11, 7, 8]


@numba.njit
def classify(types, parenttype, creaproc, edproc):
    for c in classifier:
        m = 0
        m += (c['types'] == types) or (c['types'] == 'None')
        m += (c['parenttype'] == parenttype) or (c['parenttype'] == 'None')
        m += (c['creaproc'] == creaproc) or (c['creaproc'] == 'None')
        m += (c['edproc'] == edproc) or (c['edproc'] == 'None')

        if m == 4:
            return c['A'], c['Z'], c['nestid']

    # If our data does not match any classification make it a nest None type
    # TODO: fix me
    return infinity, infinity, 12
