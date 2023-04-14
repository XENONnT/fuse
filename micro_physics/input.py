import strax
import uproot
import os
import warnings
import numba

import numpy as np
import awkward as ak

import epix

@strax.takes_config(
    strax.Option('path', default=".", track=False, infer_type=False,
                 help="Path to search for data"),
    strax.Option('file_name', track=False, infer_type=False,
                 help="File to open"),
    strax.Option('separation_scale', default=1e8, track=False, infer_type=False,
                 help="Add Description"),
    strax.Option('source_rate', default=1, track=False, infer_type=False,
                 help="source_rate"),
    strax.Option('n_interactions_per_chunk', default=10000, track=False, infer_type=False,
                 help="Add n_interactions_per_chunk"),
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
    
    #Forbid rechunking
    rechunk_on_save = False

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
            ]
    
    dtype = dtype + strax.time_fields
    
    source_done = False

    def setup(self):
        
        #Do the volume cuts here #Maybe we can move these lines somewhere else?
        self.detector_config = epix.init_detector(self.Detector.lower(), self.DetectorConfigOverride)
        outer_cylinder = getattr(epix.detectors, self.Detector.lower())
        outer_cylinder = outer_cylinder()

        self.file_reader = file_loader(self.path,
                                       self.file_name,
                                       separation_scale = self.separation_scale,
                                       event_rate = self.source_rate,
                                       n_interactions_per_chunk = self.n_interactions_per_chunk,
                                       arg_debug = self.debug,
                                       outer_cylinder=None, #This is not running 
                                       kwargs={'entry_start': self.entry_start,
                                               'entry_stop': self.entry_stop},
                                       cut_by_eventid=self.cut_by_eventid,
                                       #cut_nr_only=self.nr_only,
                                       )
        self.file_reader_iterator = self.file_reader.output_chunk()
        
    

    def compute(self):
        
        try: 
            
            chunk_data, chunk_left, chunk_right = next(self.file_reader_iterator)
            chunk_data["endtime"] = chunk_data["time"]

        except StopIteration:
            self.source_done = True
            
            chunk_left = self.file_reader.last_chunk_bounds()
            chunk_right = chunk_left + np.int64(1e4) #Add this as config option
            
            chunk_data = np.zeros(0, dtype=self.dtype)
        
        return self.chunk(start=chunk_left,
                          end=chunk_right,
                          data=chunk_data,
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
    Load the complete root file and return interactions in chunks 
    """

    def __init__(self,
                directory,
                file_name,
                separation_scale = 1e8,
                event_rate = 1,
                n_interactions_per_chunk = 500,
                cut_delayed = 4e12,
                last_chunk_length = 1e8,
                first_chunk_left = 1e6,
                chunk_delay_fraction = 0.75,
                arg_debug=False,
                outer_cylinder=None,
                kwargs={},
                cut_by_eventid=False,
                cut_nr_only=False,
                ):

        self.directory = directory
        self.file_name = file_name
        self.separation_scale = separation_scale
        self.event_rate = event_rate / 1e9 #Conversion to ns 
        self.n_interactions_per_chunk = n_interactions_per_chunk
        self.cut_delayed = cut_delayed
        self.last_chunk_length = np.int64(last_chunk_length)
        self.first_chunk_left = np.int64(first_chunk_left)
        self.chunk_delay_fraction = chunk_delay_fraction
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
    
        
        
        self.dtype = [('x', np.float32),
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
                    ]
    
        self.dtype = self.dtype + strax.time_fields
        
    
    def output_chunk(self):
        """
        Function to return one chunk of data from the root file
        """
        
        interactions, n_simulated_events, start, stop = self._load_root_file()
        
        
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
        

        inter_reshaped = self.full_array_to_numpy(interactions)
        
        #Need to check start and stop again....
        event_times = np.random.uniform(low = start/self.event_rate,
                                        high = stop/self.event_rate,
                                        size = stop-start
                                        ).astype(np.int64)
        event_times = np.sort(event_times)
        
        #Remove interactions that happen way after the run ended
        inter_reshaped = inter_reshaped[inter_reshaped["t"] < np.max(event_times) + self.cut_delayed]
        
        structure = np.unique(inter_reshaped["evtid"], return_counts = True)[1]
        
        #Check again why [:len(structure)] is needed 
        interaction_time = np.repeat(event_times[:len(structure)], structure)
        
        inter_reshaped["time"] = interaction_time + inter_reshaped["t"]
        
        #Group into chunks
        chunk_idx = dynamic_chunking(inter_reshaped["time"], scale = self.separation_scale, n_min =  self.n_interactions_per_chunk)
        
        #Calculate chunk start and end times
        chunk_start = np.array([inter_reshaped[chunk_idx == i][0]["time"] for i in np.unique(chunk_idx)])
        chunk_end = np.array([inter_reshaped[chunk_idx == i][-1]["time"] for i in np.unique(chunk_idx)])
        
        if (len(chunk_start) > 1) & (len(chunk_end) > 1):
        
            gap_length = chunk_start[1:] - chunk_end[:-1]
            gap_length = np.append(gap_length, gap_length[-1] + self.last_chunk_length)
            chunk_bounds = chunk_end + np.int64(self.chunk_delay_fraction*gap_length)
            self.chunk_bounds = np.append(chunk_start[0]-self.first_chunk_left, chunk_bounds)
            
        else: 
            print("Only one Chunk! Rate to high?")
            self.chunk_bounds = [chunk_start[0] - self.first_chunk_left, chunk_end[0]+self.last_chunk_length]
            
        #return chunk_idx, inter_reshaped, self.chunk_bounds
        
        for c_ix, chunk_left, chunk_right in zip(np.unique(chunk_idx), self.chunk_bounds[:-1], self.chunk_bounds[1:]):
            
            yield inter_reshaped[chunk_idx == c_ix], chunk_left, chunk_right
    
    def last_chunk_bounds(self):
        return self.chunk_bounds[-1]

        
    def full_array_to_numpy(self, array):
    
        len_output = len(epix.awkward_to_flat_numpy(array["x"]))

        numpy_data = np.zeros(len_output, dtype=self.dtype)

        for field in array.fields:
            numpy_data[field] = epix.awkward_to_flat_numpy(array[field])

        return numpy_data 
    
    def _load_root_file(self):
        """
        Function which reads a root file using uproot,
        performs a simple cut and builds an awkward array.
        Returns:
            interactions: awkward array
            n_simulated_events: Total number of simulated events
            start: Index of the first loaded interaction
            stop: Index of the last loaded interaction
        """

        ttree, n_simulated_events = self._get_ttree()

        if self.arg_debug:
            print(f'Total entries in input file = {ttree.num_entries}')
            cutby_string='output file entry'
            if self.cut_by_eventid:
                cutby_string='g4 eventid'

            if self.kwargs['entry_start'] is not None:
                print(f'Starting to read from {cutby_string} {self.kwargs["entry_start"]}')
            if self.kwargs['entry_stop'] is not None:
                print(f'Ending read in at {cutby_string} {self.kwargs["entry_stop"]}')

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

        # Read in data, convert mm to cm and perform a first cut if specified:
        interactions = ttree.arrays(self.column_names,
                                    self.cut_string,
                                    aliases=alias,
                                    **self.kwargs)
        eventids = ttree.arrays('eventid', **self.kwargs)
        eventids = ak.broadcast_arrays(eventids['eventid'], interactions['x'])[0]
        interactions['evtid'] = eventids

        xyz_pri = ttree.arrays(['x_pri', 'y_pri', 'z_pri'],
                              aliases={'x_pri': 'xp_pri/10',
                                       'y_pri': 'yp_pri/10',
                                       'z_pri': 'zp_pri/10'
                                      },
                              **self.kwargs)

        interactions['x_pri'] = ak.broadcast_arrays(xyz_pri['x_pri'], interactions['x'])[0]
        interactions['y_pri'] = ak.broadcast_arrays(xyz_pri['y_pri'], interactions['x'])[0]
        interactions['z_pri'] = ak.broadcast_arrays(xyz_pri['z_pri'], interactions['x'])[0]

        return interactions, n_simulated_events, start, stop
        

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
    
    
@numba.njit()
def dynamic_chunking(data, scale, n_min):

    idx_sort = np.argsort(data)
    idx_undo_sort = np.argsort(idx_sort)

    data_sorted = data[idx_sort]

    diff = data_sorted[1:] - data_sorted[:-1]

    clusters = np.array([0])
    c = 0
    for value in diff:
        if value <= scale:
            clusters = np.append(clusters, c)
            
        elif len(clusters[clusters == c]) < n_min:
            clusters = np.append(clusters, c)
            
        elif value > scale:
            c = c + 1
            clusters = np.append(clusters, c)

    clusters_undo_sort = clusters[idx_undo_sort]

    return clusters_undo_sort