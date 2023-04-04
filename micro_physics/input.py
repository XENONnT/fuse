import strax
import uproot
import os
import warnings

import numpy as np
import awkward as ak

import epix
from epix.common import awkward_to_flat_numpy, offset_range, reshape_awkward

@strax.takes_config(
    strax.Option('path', default=".", track=False, infer_type=False,
                 help="Path to search for data"),
    strax.Option('file_name', track=False, infer_type=False,
                 help="File to open"),
    strax.Option('ChunkSize', default=20, track=False, infer_type=False,
                 help="Set the number of Geant4 events in a single chunk"),
    strax.Option('debug', default=False, track=False, infer_type=False,
                 help="Show debug informations"),
    strax.Option('entry_start', default=0, track=False, infer_type=False,
                 help="First event to be read"),
    strax.Option('entry_stop', default=None, track=False, infer_type=False,
                 help="How many entries from the ROOT file you want to process. I think it is not working at the moment"),
    strax.Option('cut_by_eventid', default=False, track=False, infer_type=False,
                 help="If selected, the next two arguments act on the G4 event id, and not the entry number (default)"),
    strax.Option('nr_only', default=False, track=False, infer_type=False,
                 help="Add if you want to filter only nuclear recoil events (maximum ER energy deposit 10 keV)"),
    strax.Option('Detector', default="XENONnT", track=False, infer_type=False,
                 help="Detector to be used. Has to be defined in epix.detectors"),
    strax.Option('DetectorConfigOverride', default=None, track=False, infer_type=False,
                 help="Config file to overwrite default epix.detectors settings; see examples in the configs folder"),
)
class input_plugin(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = tuple()
    provides = "geant4_interactions"
    
    source_done = False
    
    dtype = [('x', np.float32),
             ('y', np.float32),
             ('z', np.float32),
             ('t', np.float64),
             ('ed', np.float32),
             ('type', "<U10"),
             ('trackid', np.int64),
             ('parenttype', "<U10"),
             ('parentid', np.int64),
             ('creaproc', "<U10"),
             ('edproc', "<U10"),
             ('evtid', np.int64),
             ('x_pri', np.float32),
             ('y_pri', np.float32),
             ('z_pri', np.float32),
             ('structure', np.int64),
            ]
    
    dtype = dtype + strax.time_fields
    
    source_done = False
    
    prev_chunk_stop = None
    prev_chunk_start = None

    def setup(self):
        
        #Do the volume cuts here #Maybe we can move these lines somewhere else?
        self.detector_config = epix.init_detector(self.Detector.lower(), self.DetectorConfigOverride)
        outer_cylinder = getattr(epix.detectors, self.Detector.lower())
        outer_cylinder = outer_cylinder()

        self.file_reader = file_loader(self.path,
                                       self.file_name,
                                       chunk_size = self.ChunkSize,
                                       arg_debug = self.debug,
                                       outer_cylinder=None, #This is not running 
                                       kwargs={'entry_start': self.entry_start,
                                               'entry_stop': self.entry_stop},
                                       cut_by_eventid=self.cut_by_eventid,
                                       #cut_nr_only=self.nr_only,
                                       ).load_file_in_chunks()

        
        
    
    def full_array_to_numpy(self, array):
    
        len_output = len(epix.awkward_to_flat_numpy(array["x"]))
        array_structure = np.array(epix.ak_num(array["x"]))
        array_structure = np.pad(array_structure, [0, len_output-len(array_structure)],constant_values = -1)

        numpy_data = np.zeros(len_output, dtype=self.dtype)

        for field in array.fields:
            numpy_data[field] = epix.awkward_to_flat_numpy(array[field])
        numpy_data["structure"] = array_structure
        
        return numpy_data
    

    def compute(self):
        
        try: 
            inter, n_simulated_events = next(self.file_reader)
            
            inter_reshaped = self.full_array_to_numpy(inter)
        
            inter_reshaped["time"] = (inter_reshaped["evtid"]+1) *1e9
            inter_reshaped["endtime"] = inter_reshaped["time"] +1e7
            
            
            if self.prev_chunk_stop == None:
                chunk_start = inter_reshaped['time'][0]
            else:
                chunk_start = self.prev_chunk_stop
            
            chunk_stop  = inter_reshaped['endtime'][-1]
            self.prev_chunk_stop = chunk_stop
            
            
        except StopIteration:
            self.source_done = True
            
            chunk_start = self.prev_chunk_stop
            chunk_stop = chunk_start + 1
            
            inter_reshaped = np.zeros(0, dtype=self.dtype)
        
        return self.chunk(start=chunk_start,
                          end=chunk_stop,
                          data=inter_reshaped,
                          data_type='geant4_interactions')

    
    def source_finished(self):
        return self.source_done
    
    def is_ready(self, chunk_i):
        """Overwritten to mimic online input plugin.
        Returns False to check source finished;
        Returns True to get next chunk.
        """
        if 'ready' not in self.__dict__:
            self.ready = False
        self.ready ^= True  # Flip
        return self.ready
    

class file_loader():
    """
    Class which contains functions to load geant4 interactions from
    a root file via uproot4 or interactions from a csv file via pandas.
    
    Besides loading, a simple data selection is performed. Units are
    already converted into strax conform values. mm -> cm and s -> ns.
    Args:
        directory (str): Directory in which the data is stored.
        file_name (str): File name
        arg_debug (bool): If true, print out loading information.
        outer_cylinder (dict): If specified will cut all events outside of the
            given cylinder.
        kwargs (dict): Keyword arguments passed to .arrays of
            uproot4.
        cut_by_eventid (bool): If true event start/stop are applied to
            eventids, instead of rows.
    Returns:
        awkward1.records: Interactions (eventids, parameters, types).
        integer: Number of events simulated.
    """

    def __init__(self,
                directory,
                file_name,
                chunk_size = 10,
                arg_debug=False,
                outer_cylinder=None,
                kwargs={},
                cut_by_eventid=False,
                cut_nr_only=False,
                ):

        self.directory = directory
        self.file_name = file_name
        self.chunk_size = chunk_size
        self.arg_debug = arg_debug
        self.outer_cylinder = outer_cylinder
        self.kwargs = kwargs
        self.cut_by_eventid = cut_by_eventid
        self.cut_nr_only = cut_nr_only

        self.file = os.path.join(self.directory, self.file_name)

        self.column_names = ["x", "y", "z",
                             "t", "ed",
                             "type", "trackid",
                             "parenttype", "parentid",
                             "creaproc", "edproc"]

        #Prepare cut for root and csv case
        if self.outer_cylinder:
            self.cut_string = (f'(r < {self.outer_cylinder["max_r"]})'
                               f' & ((zp >= {self.outer_cylinder["min_z"] * 10}) & (zp < {self.outer_cylinder["max_z"] * 10}))')            
        else:
            self.cut_string = None

            
    def load_file_in_chunks(self):
        #Missing: CSV file in chunks!
        
        if not self.file.endswith(".root"):
            raise ValueError(f'Cannot load events from file "{self.file}": .root file needed.')
        
        if self.arg_debug:
            print(f'Total entries in input file = {ttree.num_entries}')
            cutby_string='output file entry'
            if self.cut_by_eventid:
                cutby_string='g4 eventid'

            if self.kwargs['entry_start'] is not None:
                print(f'Starting to read from {cutby_string} {self.kwargs["entry_start"]}')
            if self.kwargs['entry_stop'] is not None:
                print(f'Ending read in at {cutby_string} {self.kwargs["entry_stop"]}')
           
        
        ttree, n_simulated_events = self._get_ttree()
        
        # If user specified entry start/stop we have to update number of
        # events for source rate computation:
        if self.kwargs['entry_start'] is not None:
            start = self.kwargs['entry_start']
        else:
            start = 0

        if self.kwargs['entry_stop'] is not None:
            stop = self.kwargs['entry_stop']
        else:
            stop = n_simulated_events
        n_simulated_events = stop - start
        
        if self.cut_by_eventid:
            # Start/stop refers to eventid so drop start drop from kwargs
            # dict if specified, otherwise we cut again on rows.
            self.kwargs.pop('entry_start', None)
            self.kwargs.pop('entry_stop', None)

        # Conversions and parameters to be computed:
        alias = {'x': 'xp/10',  # converting "geant4" mm to "straxen" cm
                 'y': 'yp/10',
                 'z': 'zp/10',
                 'r': 'sqrt(x**2 + y**2)',
                 't': 'time*10**9'
                }
        
        multi_column_iterator = uproot.iterate(ttree,
                                       self.column_names,
                                       aliases = alias,
                                       cut = self.cut_string,
                                       step_size=self.chunk_size,
                                       entrystart=start,
                                       entrystop=stop
                                      )
        event_id_iterator = uproot.iterate(ttree,
                                           ["eventid"],
                                           step_size=self.chunk_size,
                                           entrystart=start,
                                           entrystop=stop
                                          )
        pri_iterator = uproot.iterate(ttree,
                                      ['x_pri', 'y_pri', 'z_pri'],
                                      aliases = {'x_pri': 'xp_pri/10',
                                                 'y_pri': 'yp_pri/10',
                                                 'z_pri': 'zp_pri/10'},
                                      step_size=self.chunk_size,
                                      entrystart=start,
                                      entrystop=stop
                                     )

        i = 0
        for interactions, event_id, xyz_pri  in zip(multi_column_iterator,event_id_iterator, pri_iterator):
            #print(i)
            i +=1

            interactions['evtid'] = ak.broadcast_arrays(event_id["eventid"], interactions['x'])[0]

            interactions['x_pri'] = ak.broadcast_arrays(xyz_pri['x_pri'], interactions['x'])[0]
            interactions['y_pri'] = ak.broadcast_arrays(xyz_pri['y_pri'], interactions['x'])[0]
            interactions['z_pri'] = ak.broadcast_arrays(xyz_pri['z_pri'], interactions['x'])[0]

            if np.any(interactions['ed'] < 0):
                warnings.warn('At least one of the energy deposits is negative!')
            # Removing all events with zero energy deposit
            m = interactions['ed'] > 0
            if self.cut_by_eventid:
                # ufunc does not work here...
                m2 = (interactions['evtid'] >= start) & (interactions['evtid'] < stop)
                m = m & m2
            interactions = interactions[m]
            if self.cut_nr_only:
                m = ((interactions['type'] == "neutron")&(interactions['edproc'] == "hadElastic")) | (interactions['edproc'] == "ionIoni")
                e_dep_er = ak.sum(interactions[~m]['ed'], axis=1)
                e_dep_nr = ak.sum(interactions[m]['ed'], axis=1)
                interactions = interactions[(e_dep_er<10) & (e_dep_nr>0)]
            # Removing all events with no interactions:
            m = ak.num(interactions['ed']) > 0
            interactions = interactions[m]
            
            yield interactions, n_simulated_events

    def _get_ttree(self):
        """
        Function which searches for the correct ttree in MC root file.
        :param directory: Directory where file is
        :param file_name: Name of the file
        :return: root ttree and number of simulated events
        """
        root_dir = uproot.open(self.file)

        # Searching for TTree according to old/new MC file structure:
        if root_dir.classname_of('events') == 'TTree':
            ttree = root_dir['events']
            n_simulated_events = root_dir['nEVENTS'].members['fVal']
        elif root_dir.classname_of('events/events') == 'TTree':
            ttree = root_dir['events/events']
            n_simulated_events = root_dir['events/nbevents'].members['fVal']
        else:
            ttrees = []
            for k, v in root_dir.classnames().items():
                if v == 'TTree':
                    ttrees.append(k)
            raise ValueError(f'Cannot find ttree object of "{self.file}".'
                            'I tried to search in events and events/events.'
                            f'Found a ttree in {ttrees}?')
        return ttree, n_simulated_events