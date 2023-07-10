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
    
    __version__ = "0.0.0"
    
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
            log.debug("Running MergeCluster in debug mode")
        else: 
            log.setLevel('WARNING')

    def compute(self, geant4_interactions):

        if len(geant4_interactions) == 0:
            return np.zeros(0, dtype=self.dtype)

        #Use the updated clustering based only on numpy
        result = self.cluster_and_classify(geant4_interactions)

        #remove the delay cut, fuse can simulated delayed interactions! 

        result["endtime"] = result["time"]
        
        return result

    def cluster_and_classify(self, interactions):

        interaction_cluster = np.split(interactions,
                                    np.unique(interactions["cluster_ids"], return_index=True)[1][1:])

        result = np.zeros(len(interaction_cluster), dtype=self.dtype)
        
        x_avg = []
        y_avg = []
        z_avg = []
        time_avg = []
        energy_sum = []
        A_list = []
        Z_list = []
        nestid_list = []
        x_pri_list = []
        y_pri_list = []
        z_pri_list = []
        evtid_list = []

        for group in interaction_cluster:
            x_avg.append(np.average(group["x"], weights = group["ed"]))
            y_avg.append(np.average(group["y"], weights = group["ed"]))
            z_avg.append(np.average(group["z"], weights = group["ed"]))
            time_avg.append(np.average(group["time"], weights = group["ed"]))
            energy_sum.append(np.sum(group["ed"]))

            if self.tag_cluster_by == "energy":
                main_interaction_index = np.argmax(group["ed"])
            elif self.tag_cluster_by == "time":
                main_interaction_index = np.argmin(group["time"])
            else:
                raise ValueError("tag_cluster_by must be 'energy' or 'time'")

            A, Z, nestid = classify(group["type"][main_interaction_index],
                                    group["parenttype"][main_interaction_index],
                                    group["creaproc"][main_interaction_index],
                                    group["edproc"][main_interaction_index]
                                    )
            A_list.append(A)
            Z_list.append(Z)
            nestid_list.append(nestid)

            #Do i want to set them like this? What if two events are merged?
            if len(np.unique(group["evtid"])) > 1:
                log.debug("More than one event in cluster,\
                            setting primary positions and evtid based on main interaction")
            x_pri_list.append(group["x_pri"][main_interaction_index])
            y_pri_list.append(group["y_pri"][main_interaction_index])
            z_pri_list.append(group["z_pri"][main_interaction_index])
            evtid_list.append(group["evtid"][main_interaction_index])

        result["x"] = np.array(x_avg)
        result["y"] = np.array(y_avg)
        result["z"] = np.array(z_avg)
        result["time"] = np.array(time_avg)
        result["ed"] = np.array(energy_sum)
        result["A"] = np.array(A_list)
        result["Z"] = np.array(Z_list)
        result["nestid"] = np.array(nestid_list)
        result["x_pri"] = np.array(x_pri_list)
        result["y_pri"] = np.array(y_pri_list)
        result["z_pri"] = np.array(z_pri_list)
        result["evtid"] = np.array(evtid_list)

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
