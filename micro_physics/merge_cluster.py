import strax
import numpy as np
import epix
import awkward as ak
import numba


@strax.takes_config(
    strax.Option('debug', default=False, track=False, infer_type=False,
                 help="Show debug informations"),
    strax.Option('tag_cluster_by', default=False, track=False, infer_type=False,
                 help="decide if you tag the cluster (particle type, energy depositing process)\
                       according to first interaction in it (time) or most energetic (energy)"),
    strax.Option('Detector', default="XENONnT", track=False, infer_type=False,
                 help="Detector to be used. Has to be defined in epix.detectors"),
    strax.Option('DetectorConfigOverride', default=None, track=False, infer_type=False,
                 help="Config file to overwrite default epix.detectors settings; see examples in the configs folder"),
    strax.Option('max_delay', default=1e7, track=False, infer_type=False,
                 help="Time after which we cut the rest of the event (ns)"),
)
class cluster_merging(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ("geant4_interactions", "cluster_index")
    
    provides = "clustered_interactions"
    data_kind = "clustered_interactions"
    
    dtype = [('x', np.float32),
             ('y', np.float32),
             ('z', np.float32),
             ('t', np.float64),
             ('ed', np.float64),
             ('nestid', np.int64),
             ('A', np.int64),
             ('Z', np.int64),
             ('evtid', np.int64),
             ('x_pri', np.float32),
             ('y_pri', np.float32),
             ('z_pri', np.float32),
             ('xe_density', np.float32),
             ('vol_id', np.int64),
             ('create_S2', np.bool8),
             ('structure', np.int64),
            ]
    
    dtype = dtype + strax.time_fields
    
    def setup(self):
        #Do the volume cuts here #Maybe we can move these lines somewhere else?
        self.detector_config = epix.init_detector(self.Detector.lower(), self.DetectorConfigOverride)
        
    def full_array_to_numpy(self, array):
    
        len_output = len(epix.awkward_to_flat_numpy(array["x"]))
        array_structure = np.array(epix.ak_num(array["x"]))
        array_structure = np.pad(array_structure, [0, len_output-len(array_structure)],constant_values = -1)

        numpy_data = np.zeros(len_output, dtype=self.dtype)

        for field in array.fields:
            numpy_data[field] = epix.awkward_to_flat_numpy(array[field])
        numpy_data["structure"] = array_structure
        
        return numpy_data


    def compute(self, geant4_interactions):
        
        inter = ak.from_numpy(np.empty(1, dtype=geant4_interactions.dtype))
        structure = geant4_interactions["structure"][geant4_interactions["structure"]>=0]
        
        for field in inter.fields:
            inter[field] = epix.reshape_awkward(geant4_interactions[field], structure)

        result = self.cluster(inter, self.tag_cluster_by == 'energy')
        
        result['evtid'] = ak.broadcast_arrays(inter['evtid'][:, 0], result['ed'])[0]
        # Add x_pri, y_pri, z_pri again:
        result['x_pri'] = ak.broadcast_arrays(inter['x_pri'][:, 0], result['ed'])[0]
        result['y_pri'] = ak.broadcast_arrays(inter['y_pri'][:, 0], result['ed'])[0]
        result['z_pri'] = ak.broadcast_arrays(inter['z_pri'][:, 0], result['ed'])[0]
        
        
        res_det = epix.in_sensitive_volume(result, self.detector_config)
        for field in res_det.fields:
            result[field] = res_det[field]
        m = result['vol_id'] > 0  # All volumes have an id larger zero
        result = result[m]
        
        # Removing now empty events as a result of the selection above:
        m = epix.ak_num(result['ed']) > 0
        result = result[m]
        
        # Sort entries (in an event) by in time, then chop all delayed
        # events which are too far away from the rest.
        # (This is a requirement of WFSim)
        result = result[ak.argsort(result['t'])]
        dt = epix.calc_dt(result)
        result = result[dt <= self.max_delay]
        
        result = self.full_array_to_numpy(result)
        
        result["time"] = (result["evtid"]+1) *1e9
        result["endtime"] = result["time"] +1e7
        
        return result
    
    def cluster(self, inter, classify_by_energy=False):
        """
        Function which clusters the found clusters together.
        To cluster events a weighted mean is computed for time and position.
        The individual interactions are weighted by their energy.
        The energy of clustered interaction is given by the total sum.
        Events can be classified either by the first interaction in time in the
        cluster or by the highest energy deposition.
        Args:
            inter (awkward.Array): Array containing at least the following
                fields: x,y,z,t,ed,cluster_ids, type, parenttype, creaproc,
                edproc.
        Kwargs:
            classify_by_energy (bool): If true events are classified
                according to the properties of the highest energy deposit
                within the cluster. If false cluster is classified according
                to first interaction.
        Returns:
            awkward.Array: Clustered events with nest conform
                classification.
        """

        if len(inter) == 0:
            result_cluster_dtype = [('x', 'float64'),
                                    ('y', 'float64'),
                                    ('z', 'float64'),
                                    ('t', 'float64'),
                                    ('ed', 'float64'),
                                    ('nestid', 'int64'),
                                    ('A', 'int64'),
                                    ('Z', 'int64')]
            return ak.from_numpy(np.empty(0, dtype=result_cluster_dtype))
        # Sort interactions by cluster_ids to simplify looping
        inds = ak.argsort(inter['cluster_ids'])
        inter = inter[inds]

        # TODO: Better way to do this with awkward?
        x = inter['x']
        y = inter['y']
        z = inter['z']
        ed = inter['ed']
        time = inter['t']
        ci = inter['cluster_ids']
        types = inter['type']
        parenttype = inter['parenttype']
        creaproc = inter['creaproc']
        edproc = inter['edproc']

        # Init result and cluster:
        res = ak.ArrayBuilder()
        _cluster(x, y, z, ed, time, ci,
                 types, parenttype, creaproc, edproc,
                 classify_by_energy, res)
        return res.snapshot()


@numba.njit
def _cluster(x, y, z, ed, time, ci,
             types, parenttype, creaproc, edproc,
             classify_by_energy, res):
    # Loop over each event
    nevents = len(ed)
    for ei in range(nevents):
        # Init a new list for clustered interactions within event:
        res.begin_list()

        # Init buffers:
        ninteractions = len(ed[ei])
        x_mean = 0
        y_mean = 0
        z_mean = 0
        t_mean = 0
        ed_tot = 0
        event_time_min = min(time[ei])

        current_ci = 0  # Current cluster id
        i_class = 0  # Index for classification (depends on users requirement)
        # Set classifier start value according to user request, interactions
        # are classified either by
        if classify_by_energy:
            # Highest energy
            classifier_max = 0
        else:
            # First interaction
            classifier_max = np.inf

        # Loop over all interactions within event:
        for ii in range(ninteractions):
            if current_ci != ci[ei][ii]:
                # Cluster Id has changed compared to previous interaction,
                # hence we have to write out our result and empty the buffer,
                # but first classify event:
                A, Z, nestid = classify(types[ei][i_class],
                                        parenttype[ei][i_class],
                                        creaproc[ei][i_class],
                                        edproc[ei][i_class])

                # Write result, simple but extensive with awkward...
                _write_result(res, x_mean, y_mean, z_mean,
                              ed_tot, t_mean, event_time_min, A, Z, nestid)

                # Update cluster id and empty buffer
                current_ci = ci[ei][ii]
                x_mean = 0
                y_mean = 0
                z_mean = 0
                t_mean = 0
                ed_tot = 0

                # Reset classifier:
                if classify_by_energy:
                    classifier_max = 0
                else:
                    classifier_max = np.inf

            # We have to gather information of current cluster:
            e = ed[ei][ii]
            t = time[ei][ii] - event_time_min
            x_mean += x[ei][ii] * e
            y_mean += y[ei][ii] * e
            z_mean += z[ei][ii] * e
            t_mean += t * e
            ed_tot += e

            if classify_by_energy:
                # In case we want to classify the event by energy.
                if e > classifier_max:
                    i_class = ii
                    classifier_max = e
            else:
                # or by first arrival time:
                if t < classifier_max:
                    i_class = ii
                    classifier_max = t

        # Before we are done with this event we have to classify and
        # write the last interaction
        A, Z, nestid = classify(types[ei][i_class],
                                parenttype[ei][i_class],
                                creaproc[ei][i_class],
                                edproc[ei][i_class])

        _write_result(res, x_mean, y_mean, z_mean,
                      ed_tot, t_mean, event_time_min, A, Z, nestid)

        res.end_list()


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


@numba.njit
def _write_result(res, x_mean, y_mean, z_mean,
                  ed_tot, t_mean, event_time_min, A, Z, nestid):
    """
    Helper to write result into record array.
    """
    res.begin_record()
    res.field('x')
    res.real(x_mean / ed_tot)
    res.field('y')
    res.real(y_mean / ed_tot)
    res.field('z')
    res.real(z_mean / ed_tot)
    res.field('t')
    res.real((t_mean / ed_tot) + event_time_min)
    res.field('ed')
    res.real(ed_tot)
    res.field('nestid')
    res.integer(nestid)
    res.field('A')
    res.integer(A)
    res.field('Z')
    res.integer(Z)
    res.end_record()
        