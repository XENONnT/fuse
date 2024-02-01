import strax
import straxen
import uproot
import os
import logging

import pandas as pd
import numpy as np
import awkward as ak

export, __all__ = strax.exporter()

from ...common import full_array_to_numpy, reshape_awkward, dynamic_chunking
from ...plugin import FuseBasePlugin

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.micro_physics.input')

#Remove the path and file name option from the config and do this with the run_number??
@export
class ChunkInput(FuseBasePlugin):
    
    __version__ = "0.1.2"
    
    depends_on = tuple()
    provides = "geant4_interactions"

    source_done = False
    
    dtype = [('x', np.float64),
             ('y', np.float64),
             ('z', np.float64),
             ('t', np.float64),
             ('ed', np.float32),
             ('type', "<U10"),
             ('trackid', np.int16),
             ('parenttype', "<U10"),
             ('parentid', np.int16),
             ('creaproc', "<U10"),
             ('edproc', "<U10"),
             ('evtid', np.int32),
             ('x_pri', np.float32),
             ('y_pri', np.float32),
             ('z_pri', np.float32),
            ]
    
    dtype = dtype + strax.time_fields

    save_when = strax.SaveWhen.TARGET
    
    source_done = False

    #Config options
    path = straxen.URLConfig(
        track=False,
        help='Path to the file to simulate from excluding the file name',
    )

    file_name = straxen.URLConfig(
        track=False,
        help='Name of the file to simulate from',
    )

    separation_scale = straxen.URLConfig(
        default=1e8, type=(int, float),
        help='Separation scale for the dynamic chunking in ns',
    )

    source_rate = straxen.URLConfig(
        default=1, type=(int, float),
        help='Source rate used to generate event times'
             'Use a value >0 to generate event times in fuse'
             'Use source_rate = 0 to use event times from the input file (only for csv input)',
    )

    cut_delayed = straxen.URLConfig(
        default=9e18, type=(int, float),
        help='All interactions happening after this time (including the event time) will be cut.',
    )

    n_interactions_per_chunk = straxen.URLConfig(
        default=1e5, type=(int, float),
        help='Minimum number of interaction per chunk',
    )

    entry_start = straxen.URLConfig(
        default=0, type=(int, float),
        help='Geant4 event to start simulation from.',
    )

    entry_stop = straxen.URLConfig(
        default=None,
        help='Geant4 event to stop simulation at. If None, all events are simulated.',
    )

    cut_by_eventid = straxen.URLConfig(
        default=False, type=bool,
        help='If selected, the next two arguments act on the G4 event id, and not the entry number (default)',
    )

    nr_only = straxen.URLConfig(
        default=False, type=bool,
        help='Filter only nuclear recoil events (maximum ER energy deposit 10 keV)',
    )

    def setup(self):
        super().setup()
       
        self.file_reader = file_loader(self.path,
                                       self.file_name,
                                       self.rng,
                                       separation_scale = self.separation_scale,
                                       event_rate = self.source_rate,
                                       cut_delayed = self.cut_delayed,
                                       n_interactions_per_chunk = self.n_interactions_per_chunk,
                                       arg_debug = self.debug,
                                       outer_cylinder=None, #This is not running 
                                       kwargs={'entry_start': self.entry_start,
                                               'entry_stop': self.entry_stop},
                                       cut_by_eventid=self.cut_by_eventid,
                                       cut_nr_only=self.nr_only,
                                       )
        self.file_reader_iterator = self.file_reader.output_chunk()
        
    def compute(self):
        
        try: 
            chunk_data, chunk_left, chunk_right, source_done = next(self.file_reader_iterator)
            chunk_data["endtime"] = chunk_data["time"]

            self.source_done = source_done

            return self.chunk(start=chunk_left,
                              end=chunk_right,
                              data=chunk_data,
                              data_type='geant4_interactions'
                              )

        except StopIteration:
            raise RuntimeError("Bug in chunk building!")

    
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
                random_number_generator,
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
        self.rng = random_number_generator
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
    
        
        
        self.dtype = [('x', np.float64),
                     ('y', np.float64),
                     ('z', np.float64),
                     ('t', np.float64),
                     ('ed', np.float32),
                     ('type', "<U10"),
                     ('trackid', np.int16),
                     ('parenttype', "<U10"),
                     ('parentid', np.int16),
                     ('creaproc', "<U10"),
                     ('edproc', "<U10"),
                     ('evtid', np.int32),
                     ('x_pri', np.float32),
                     ('y_pri', np.float32),
                     ('z_pri', np.float32),
                    ]
    
        self.dtype = self.dtype + strax.time_fields
        
    
    def output_chunk(self):
        """
        Function to return one chunk of data from the root file
        """
        
        if self.file.endswith(".root"):
            interactions, n_simulated_events, start, stop = self._load_root_file()
        elif self.file.endswith(".csv"):
            interactions, n_simulated_events, start, stop = self._load_csv_file()
        else:
            raise ValueError(f'Cannot load events from file "{self.file}": .root or .cvs file needed.')        
        
        # Removing all events with zero energy deposit
        m = interactions['ed'] > 0
        if self.cut_by_eventid:
            # ufunc does not work here...
            m2 = (interactions['evtid'] >= start) & (interactions['evtid'] < stop)
            m = m & m2
        interactions = interactions[m]

        if self.cut_nr_only:
            log.info("'nr_only' set to True, keeping only the NR events")
            m = ((interactions['type'] == "neutron")&(interactions['edproc'] == "hadElastic")) | (interactions['edproc'] == "ionIoni")
            e_dep_er = ak.sum(interactions[~m]['ed'], axis=1)
            e_dep_nr = ak.sum(interactions[m]['ed'], axis=1)
            interactions = interactions[(e_dep_er<10) & (e_dep_nr>0)]

        # Removing all events with no interactions:
        m = ak.num(interactions['ed']) > 0
        interactions = interactions[m]

        #Sort interactions in events by time and subtract time of the first interaction
        interactions = interactions[ak.argsort(interactions['t'])]
        interactions['t'] = interactions['t'] - interactions['t'][:, 0]

        inter_reshaped = full_array_to_numpy(interactions, self.dtype)
        
        #Need to check start and stop again....
        if self.event_rate > 0:
            event_times = self.rng.uniform(low = start/self.event_rate,
                                            high = stop/self.event_rate,
                                            size = stop-start
                                            ).astype(np.int64)
            event_times = np.sort(event_times)

            structure = np.unique(inter_reshaped["evtid"], return_counts = True)[1]

            #Check again why [:len(structure)] is needed 
            interaction_time = np.repeat(event_times[:len(structure)], structure)
            inter_reshaped["time"] = interaction_time + inter_reshaped["t"]
        elif self.event_rate == 0:
            log.info("Using event times from provided input file.")
            if self.file.endswith(".root"):
                log.warning("Using event times from root file is not recommended! Use a source_rate > 0 instead.")
            inter_reshaped["time"] = inter_reshaped["t"]
        else:
            raise ValueError("Source rate cannot be negative!")
        
        #Remove interactions that happen way after the run ended
        delay_cut = inter_reshaped["t"] <= self.cut_delayed
        log.info(f"Removing {np.sum(~delay_cut)} ( {np.sum(~delay_cut)/len(delay_cut) *100:.4} %) interactions later than {self.cut_delayed:.2e} ns.")
        inter_reshaped = inter_reshaped[delay_cut]
 
        sort_idx = np.argsort(inter_reshaped["time"])
        inter_reshaped = inter_reshaped[sort_idx]

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
            log.warning(
                "Only one Chunk created! Only a few events simulated? If no, your chunking parameters might not be optimal. "
                "Try to decrease the source_rate or decrease the n_interactions_per_chunk."
            )
            self.chunk_bounds = [chunk_start[0] - self.first_chunk_left, chunk_end[0]+self.last_chunk_length]
        
        source_done = False
        unique_chunk_index_values = np.unique(chunk_idx)
        log.info(f"Simulating data in {len(unique_chunk_index_values)} chunks.")
        for c_ix, chunk_left, chunk_right in zip(unique_chunk_index_values, self.chunk_bounds[:-1], self.chunk_bounds[1:]):
            
            if c_ix == unique_chunk_index_values[-1]:
                source_done = True
                log.debug("Last chunk created!")

            yield inter_reshaped[chunk_idx == c_ix], chunk_left, chunk_right, source_done
    
    def last_chunk_bounds(self):
        return self.chunk_bounds[-1]

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
            log.info(f'Total entries in input file = {ttree.num_entries}')
            cutby_string='output file entry'
            if self.cut_by_eventid:
                cutby_string='g4 eventid'

            if self.kwargs['entry_start'] is not None:
                log.debug(f'Starting to read from {cutby_string} {self.kwargs["entry_start"]}')
            if self.kwargs['entry_stop'] is not None:
                log.debug(f'Ending read in at {cutby_string} {self.kwargs["entry_stop"]}')
           
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
    
    def _load_csv_file(self):
        """ 
        Function which reads a csv file using pandas, 
        performs a simple cut and builds an awkward array.

        Returns:
            interactions: awkward array
            n_simulated_events: Total number of simulated events
            start: Index of the first loaded interaction
            stop: Index of the last loaded interaction
        """

        log.debug("Load instructions from a csv file!")
        
        instr_df =  pd.read_csv(self.file)

        #unit conversion similar to root case
        instr_df["x"] = instr_df["xp"]/10 
        instr_df["y"] = instr_df["yp"]/10 
        instr_df["z"] = instr_df["zp"]/10
        instr_df["x_pri"] = instr_df["xp_pri"]/10
        instr_df["y_pri"] = instr_df["yp_pri"]/10
        instr_df["z_pri"] = instr_df["zp_pri"]/10
        instr_df["r"] = np.sqrt(instr_df["x"]**2 + instr_df["y"]**2)
        instr_df["t"] = instr_df["time"]*10**9

        #Check if all needed columns are in place:
        if not set(self.column_names).issubset(instr_df.columns):
            log.warning("Not all needed columns provided!")

        n_simulated_events = len(np.unique(instr_df.eventid))

        if self.outer_cylinder:
            instr_df = instr_df.query(self.cut_string)
            
        instr_df = instr_df[self.column_names+["eventid", "x_pri", "y_pri", "z_pri"]]

        interactions = self._awkwardify_df(instr_df)

        #Use always all events in the csv file
        start = 0
        stop = n_simulated_events

        return interactions, n_simulated_events, start, stop 
    
    @staticmethod
    def _awkwardify_df(df):
        """
        Function which builds an jagged awkward array from pandas dataframe.

        Args:
            df: Pandas Dataframe

        Returns:
            ak.Array(dictionary): awkward array

        """

        _, evt_offsets = np.unique(df["eventid"], return_counts = True)
    
        dictionary = {"x": reshape_awkward(df["x"].values , evt_offsets),
                      "y": reshape_awkward(df["y"].values , evt_offsets),
                      "z": reshape_awkward(df["z"].values , evt_offsets),
                      "x_pri": reshape_awkward(df["x_pri"].values, evt_offsets),
                      "y_pri": reshape_awkward(df["y_pri"].values, evt_offsets),
                      "z_pri": reshape_awkward(df["z_pri"].values, evt_offsets),
                      "t": reshape_awkward(df["t"].values , evt_offsets),
                      "ed": reshape_awkward(df["ed"].values , evt_offsets),
                      "type":reshape_awkward(np.array(df["type"], dtype=str) , evt_offsets),
                      "trackid": reshape_awkward(df["trackid"].values , evt_offsets),
                      "parenttype": reshape_awkward(np.array(df["parenttype"], dtype=str) , evt_offsets),
                      "parentid": reshape_awkward(df["parentid"].values , evt_offsets),
                      "creaproc": reshape_awkward(np.array(df["creaproc"], dtype=str) , evt_offsets),
                      "edproc": reshape_awkward(np.array(df["edproc"], dtype=str) , evt_offsets),
                      "evtid": reshape_awkward(df["eventid"].values , evt_offsets),
                    }

        return ak.Array(dictionary)