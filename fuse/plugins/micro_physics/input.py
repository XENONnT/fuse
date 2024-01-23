import strax
import straxen
import uproot
import os
import logging

import pandas as pd
import numpy as np
import awkward as ak

export, __all__ = strax.exporter()

from ...common import full_array_to_numpy, reshape_awkward, dynamic_chunking, FUSE_PLUGIN_TIMEOUT

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.micro_physics.input')

#Remove the path and file name option from the config and do this with the run_number??
@export
class ChunkInput(strax.Plugin):
    
    __version__ = "0.2.0"
    
    depends_on = tuple()
    provides = ("geant4_interactions", "nv_pmthits")
    
    #Forbid rechunking
    rechunk_on_save = False

    source_done = False

    input_timeout = FUSE_PLUGIN_TIMEOUT
    
    dtype_tpc = [('x', np.float64),
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
    
    dtype_tpc = dtype_tpc + strax.time_fields


    dtype_nv = [('pmthitEnergy', np.float32),
                ('pmthitTime', np.float32),
                ('pmthitID', np.int16),
                ('evtid', np.int32),
                ]
    dtype_nv = dtype_nv + strax.time_fields

    dtype = dict()
    dtype["geant4_interactions"] = dtype_tpc
    dtype["nv_pmthits"] = dtype_nv

    data_kind = {"geant4_interactions": "geant4_interactions",
                 "nv_pmthits" : "nv_pmthits"
                }


    save_when = strax.SaveWhen.TARGET
    
    source_done = False

    #Config options
    debug = straxen.URLConfig(
        default=False, type=bool,track=False,
        help='Show debug information during simulation',
    )

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

    file_size_limit = straxen.URLConfig(
        default=500, type=(int, float),
        help='Target file size limit in MB',
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

    nv_output = straxen.URLConfig(
        default=True, type=bool,
        help='Decide if you want to have NV output or not',
    )

    deterministic_seed = straxen.URLConfig(
        default=True, type=bool,
        help='Set the random seed from lineage and run_id, or pull the seed from the OS.',
    )

    def setup(self):

        if self.debug:
            log.setLevel('DEBUG')
            log.debug(f"Running ChunkInput version {self.__version__} in debug mode")
        else:
            log.setLevel('INFO')

        if self.deterministic_seed:
            hash_string = strax.deterministic_hash((self.run_id, self.lineage))
            seed = int(hash_string.encode().hex(), 16)
            self.rng = np.random.default_rng(seed = seed)
            log.debug(f"Generating random numbers from seed {seed}")
        else: 
            self.rng = np.random.default_rng()
            log.debug(f"Generating random numbers with seed pulled from OS")

        self.file_reader = file_loader(self.path,
                                       self.file_name,
                                       self.rng,
                                       separation_scale = self.separation_scale,
                                       event_rate = self.source_rate,
                                       cut_delayed = self.cut_delayed,
                                       file_size_limit = self.file_size_limit,
                                       arg_debug = self.debug,
                                       outer_cylinder=None, #This is not running 
                                       kwargs={'entry_start': self.entry_start,
                                               'entry_stop': self.entry_stop},
                                       cut_by_eventid=self.cut_by_eventid,
                                       cut_nr_only=self.nr_only,
                                       neutron_veto_output=self.nv_output,
                                       )
        self.file_reader_iterator = self.file_reader.output_chunk()
        
    def compute(self):
        
        try: 
            
            if self.nv_output:
                
                chunk_data_tpc, chunk_left, chunk_right, source_done, chunk_data_nv = next(self.file_reader_iterator)
                chunk_data_tpc["endtime"] = chunk_data_tpc["time"]
                chunk_data_nv["endtime"] = chunk_data_nv["time"]

            else:
                chunk_data_tpc, chunk_left, chunk_right, source_done = next(self.file_reader_iterator)
                chunk_data_tpc["endtime"] = chunk_data_tpc["time"]

                chunk_data_nv = np.zeros(0, dtype=self.dtype["nv_pmthits"])

            self.source_done = source_done

            result = {"geant4_interactions" : self.chunk(start=chunk_left, end=chunk_right, data=chunk_data_tpc, data_type = "geant4_interactions"),
                      "nv_pmthits" : self.chunk(start=chunk_left, end=chunk_right, data=chunk_data_nv, data_type = "nv_pmthits")
                      }


            return result

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
                file_size_limit = 500,
                cut_delayed = 4e12,
                last_chunk_length = 1e8,
                first_chunk_left = 1e6,
                chunk_delay_fraction = 0.75,
                arg_debug=False,
                outer_cylinder=None,
                kwargs={},
                cut_by_eventid=False,
                cut_nr_only=False,
                neutron_veto_output=False,
                ):

        self.directory = directory
        self.file_name = file_name
        self.rng = random_number_generator
        self.separation_scale = separation_scale
        self.event_rate = event_rate / 1e9 #Conversion to ns 
        self.file_size_limit = file_size_limit
        self.cut_delayed = cut_delayed
        self.last_chunk_length = np.int64(last_chunk_length)
        self.first_chunk_left = np.int64(first_chunk_left)
        self.chunk_delay_fraction = chunk_delay_fraction
        self.arg_debug = arg_debug
        self.outer_cylinder = outer_cylinder
        self.kwargs = kwargs
        self.cut_by_eventid = cut_by_eventid
        self.cut_nr_only = cut_nr_only
        self.neutron_veto_output = neutron_veto_output
        
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

        if self.neutron_veto_output:
            self.nv_dtype = [('pmthitEnergy', np.float32),
                             ('pmthitTime', np.float32),
                             ('pmthitID', np.int16),
                             ('evtid', np.int32),
                            ]
            self.nv_dtype = self.nv_dtype + strax.time_fields

            self.neutron_veto_column_names = ['pmthitEnergy','pmthitTime','pmthitID']
        
    
    def output_chunk(self):
        """
        Function to return one chunk of data from the root file
        """
        
        if self.file.endswith(".root"):
            if self.neutron_veto_output:
                interactions, n_simulated_events, start, stop, nv_interactions = self._load_root_file()
            else:
                interactions, n_simulated_events, start, stop = self._load_root_file()
        elif self.file.endswith(".csv"):
            interactions, n_simulated_events, start, stop = self._load_csv_file()
        else:
            raise ValueError(f'Cannot load events from file "{self.file}": .root or .cvs file needed.')        
        
        # Removing all events with zero energy deposit
        #m = interactions['ed'] > 0
        #if self.cut_by_eventid:
            # ufunc does not work here...
        #    m2 = (interactions['evtid'] >= start) & (interactions['evtid'] < stop)
        #    m = m & m2

        #if self.neutron_veto_output:
        #    m_nv = nv_interactions['pmthitEnergy'] > 0
            #if self.cut_by_eventid:
            #    m_nv = m_nv & m2
        #    nv_interactions = nv_interactions[m_nv]
        
        #interactions = interactions[m]

        #if self.cut_nr_only:
        #    log.info("'nr_only' set to True, keeping only the NR events")
        #    m = ((interactions['type'] == "neutron")&(interactions['edproc'] == "hadElastic")) | (interactions['edproc'] == "ionIoni")
        #    e_dep_er = ak.sum(interactions[~m]['ed'], axis=1)
        #    e_dep_nr = ak.sum(interactions[m]['ed'], axis=1)
        #    interactions = interactions[(e_dep_er<10) & (e_dep_nr>0)]

        # Removing all events with no interactions:
        #m = ak.num(interactions['ed']) > 0
        #interactions = interactions[m]

        #Sort interactions in events by time and subtract time of the first interaction
        
        min_time_TPC = ak.min(interactions['t'], axis=1)
        if self.neutron_veto_output:
            min_time_NV = ak.min(nv_interactions['pmthitTime'], axis=1)
            global_min_time = ak.min([min_time_TPC, min_time_NV], axis=0)
        else:
            global_min_time = min_time_TPC

        interactions = interactions[ak.argsort(interactions['t'])]
        interactions['t'] = interactions['t'] - global_min_time

        inter_reshaped = full_array_to_numpy(interactions, self.dtype)

        if self.neutron_veto_output:
            #m = ak.num(nv_interactions['pmthitEnergy']) > 0
            #nv_interactions = nv_interactions[m]
            nv_interactions = nv_interactions[ak.argsort(nv_interactions['pmthitTime'])]
            nv_interactions['pmthitTime'] = nv_interactions['pmthitTime'] - global_min_time
            nv_inter_reshaped = full_array_to_numpy(nv_interactions, self.nv_dtype)

        if self.event_rate > 0:
            event_times = self.rng.uniform(low = start/self.event_rate,
                                            high = stop/self.event_rate,
                                            size = stop-start
                                            ).astype(np.int64)
            event_times = np.sort(event_times)

            structure, counts = np.unique(inter_reshaped["evtid"], return_counts = True)

            interaction_time = np.repeat(event_times[structure], counts)
            inter_reshaped["time"] = interaction_time + inter_reshaped["t"]

            if self.neutron_veto_output:
                structure_nv, counts_nv = np.unique(nv_inter_reshaped["evtid"], return_counts = True)
                interaction_time_nv = np.repeat(event_times[structure_nv], counts_nv)
                #Apply the same event times to the neutron veto output
                nv_inter_reshaped["time"] = interaction_time_nv + nv_inter_reshaped["pmthitTime"]

        elif self.event_rate == 0:
            log.info("Using event times from provided input file.")
            if self.file.endswith(".root"):
                log.warning("Using event times from root file is not recommended! Use a source_rate > 0 instead.")
            inter_reshaped["time"] = inter_reshaped["t"]
            if self.neutron_veto_output:
                nv_inter_reshaped["time"] = nv_inter_reshaped["pmthitTime"]
        else:
            raise ValueError("Source rate cannot be negative!")
        
        #Remove interactions that happen way after the run ended
        delay_cut = inter_reshaped["t"] <= self.cut_delayed
        log.info(f"Removing {np.sum(~delay_cut)} ( {np.sum(~delay_cut)/len(delay_cut) *100:.4} %) interactions later than {self.cut_delayed:.2e} ns.")
        inter_reshaped = inter_reshaped[delay_cut]
        if self.neutron_veto_output:
            nv_delay_cut = nv_inter_reshaped["pmthitTime"] <= self.cut_delayed
            log.info(f"Removing {np.sum(~nv_delay_cut)} ( {np.sum(~nv_delay_cut)/len(nv_delay_cut) *100:.4} %) NV interactions later than {self.cut_delayed:.2e} ns.")
            nv_inter_reshaped = nv_inter_reshaped[nv_delay_cut]
 
        sort_idx = np.argsort(inter_reshaped["time"])
        inter_reshaped = inter_reshaped[sort_idx]
        if self.neutron_veto_output:
            nv_sort_idx = np.argsort(nv_inter_reshaped["time"])
            nv_inter_reshaped = nv_inter_reshaped[nv_sort_idx]

        #Group into chunks

        #Move this somewhere else?
        n_bytes_per_interaction_TPC = 6*8 + 5*4 + 2*2 + 4*40
        n_bytes_per_interaction_NV = 2*8 + 3*4 + 1*2

        if self.neutron_veto_output:
            #Assume that both sets are already sorted!
            combined_times = np.append(inter_reshaped["time"], nv_inter_reshaped["time"])
            combined_types = np.append(np.zeros(len(inter_reshaped)), np.ones(len(nv_inter_reshaped)))

            sort_idx = np.argsort(combined_times)
            combined_times = combined_times[sort_idx]
            combined_types = combined_types[sort_idx]
            combined_time_gaps = combined_times[1:] - combined_times[:-1]
            combined_time_gaps = np.append(combined_time_gaps, 0) #Add last gap

            tpc_split_index, nv_split_index = dynamic_chunking_two_outputs(combined_time_gaps,
                                                                           combined_types,
                                                                           self.file_size_limit,
                                                                           self.separation_scale,
                                                                           n_bytes_per_interaction_TPC,
                                                                           n_bytes_per_interaction_NV,
                                                                         )
            inter_reshaped_chunks = np.array_split(inter_reshaped, tpc_split_index)
            nv_inter_reshaped_chunks = np.array_split(nv_inter_reshaped, nv_split_index)

            #Calculate chunk start and end times
            chunk_start_tpc = []
            chunk_end_tpc = []
            chunk_start_nv = []
            chunk_end_nv = []
            for i in range(len(inter_reshaped_chunks)):
            
                if len(inter_reshaped_chunks[i]) == 0:
                    chunk_start_tpc.append(np.inf)
                    chunk_end_tpc.append(-np.inf)
                else:
                    chunk_start_tpc.append(inter_reshaped_chunks[i][0]["time"])
                    chunk_end_tpc.append(inter_reshaped_chunks[i][-1]["time"])
            
                if len(nv_inter_reshaped_chunks[i]) == 0:
                    chunk_start_nv.append(np.inf)
                    chunk_end_nv.append(-np.inf)
                else:
                    chunk_start_nv.append(nv_inter_reshaped_chunks[i][0]["time"])
                    chunk_end_nv.append(nv_inter_reshaped_chunks[i][-1]["time"])
            chunk_start_tpc = np.array(chunk_start_tpc)
            chunk_end_tpc = np.array(chunk_end_tpc)
            chunk_start_nv = np.array(chunk_start_nv)
            chunk_end_nv = np.array(chunk_end_nv)

            chunk_start, chunk_end = combine_chunk_times(chunk_start_tpc, chunk_end_tpc, chunk_start_nv, chunk_end_nv)

        else:
            #The only TPC case:

            time_gaps = inter_reshaped["time"][1:] - inter_reshaped["time"][:-1] 
            time_gaps = np.append(time_gaps, 0) #Add last gap

            split_index = dynamic_chunking(time_gaps,
                                           self.file_size_limit,
                                           self.separation_scale,
                                           n_bytes_per_interaction_TPC,
                                           )
            inter_reshaped_chunks = np.array_split(inter_reshaped, split_index)

            chunk_start_tpc = []
            chunk_end_tpc = []
            for i in range(len(inter_reshaped_chunks)):
                chunk_start_tpc.append(inter_reshaped_chunks[i][0]["time"])
                chunk_end_tpc.append(inter_reshaped_chunks[i][-1]["time"])
            chunk_start = np.array(chunk_start_tpc)
            chunk_end = np.array(chunk_end_tpc)

        if (len(chunk_start) > 1) & (len(chunk_end) > 1):
        
            gap_length = chunk_start[1:] - chunk_end[:-1]
            gap_length = np.append(gap_length, gap_length[-1] + self.last_chunk_length)
            chunk_bounds = chunk_end + np.int64(self.chunk_delay_fraction*gap_length)
            self.chunk_bounds = np.append(chunk_start[0]-self.first_chunk_left, chunk_bounds)
            
        else: 
            log.warning("Only one Chunk created! Only a few events simulated? If no, your chunking parameters might not be optimal.")
            log.warning("Try to decrease the source_rate or decrease the n_interactions_per_chunk")
            self.chunk_bounds = [chunk_start[0] - self.first_chunk_left, chunk_end[0]+self.last_chunk_length]

        source_done = False
        number_of_chunks = len(self.chunk_bounds)-1
        log.info(f"Simulating data in {number_of_chunks} chunks.")
        for c_ix, (chunk_left, chunk_right) in enumerate(zip(self.chunk_bounds[:-1], self.chunk_bounds[1:])):

            if c_ix == number_of_chunks-1:
                source_done = True
                log.debug("Last chunk created!")
            if self.neutron_veto_output:
                yield inter_reshaped_chunks[c_ix], chunk_left, chunk_right, source_done, nv_inter_reshaped_chunks[c_ix]
            else:
                yield inter_reshaped_chunks[c_ix], chunk_left, chunk_right, source_done
    
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

        #If we want to have NV output: 
        if self.neutron_veto_output:
            
            #Do we need this?
            #nv_alias = {'pmthitTime': 'pmthitTime*10**9'}

            nv_interactions = ttree.arrays(self.neutron_veto_column_names,
                                           #aliases=nv_alias,
                                           **self.kwargs)
            nv_interactions['pmthitTime'] = nv_interactions['pmthitTime']*10**9                              
            nv_eventids = ttree.arrays('eventid', **self.kwargs)
            nv_eventids = ak.broadcast_arrays(nv_eventids['eventid'], nv_interactions['pmthitTime'])[0]
            nv_interactions['evtid'] = nv_eventids

            return interactions, n_simulated_events, start, stop, nv_interactions

        else:

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


@njit(cache=True)
def dynamic_chunking(time_gaps,
                     file_size_limit,
                     min_gap_length,
                     n_bytes_per_interaction,
                     ):

    data_size_mb = 0
    split_index = []

    running_index = 0

    for g in time_gaps:
        
        data_size_mb += n_bytes_per_interaction / 1e6
        running_index += 1

        if data_size_mb < file_size_limit: 
            continue

        if g >= min_gap_length:
            data_size_mb = 0
            split_index.append(running_index)

    return np.array(split_index)



@njit(cache=True)
def dynamic_chunking_two_outputs(combined_time_gaps,
                                 combined_types,
                                 file_size_limit,
                                 min_gap_length,
                                 n_bytes_per_interaction_TPC,
                                 n_bytes_per_interaction_NV,
                                 ):
    """Function to split the TPC and NV data into chunks based on the time gaps between the interactions"""

    data_size_mb_tpc = 0
    data_size_mb_nv = 0
    split_index_tpc = []
    split_index_nv = []

    running_index_tpc = 0
    running_index_nv = 0

    for i, (t, g) in enumerate(zip(combined_types, combined_time_gaps)):
        
        if t == 0:
            data_size_mb_tpc += n_bytes_per_interaction_TPC / 1e6
            running_index_tpc += 1
        elif t == 1:
            data_size_mb_nv += n_bytes_per_interaction_NV / 1e6
            running_index_nv += 1

        if (data_size_mb_tpc < file_size_limit) & (data_size_mb_nv < file_size_limit): 
            continue

        if g >= min_gap_length:
            data_size_mb_tpc = data_size_mb_nv = 0
            split_index_tpc.append(running_index_tpc)
            split_index_nv.append(running_index_nv)

    return np.array(split_index_tpc), np.array(split_index_nv)

def combine_chunk_times(cs_tpc, ce_tpc, cs_nv, ce_nv):
    """Function to combine TPC and NV chunk times"""
    combined_cs = []

    number_of_chunks = len(cs_tpc)

    for i in range(number_of_chunks):
        if cs_tpc[i] < cs_nv[i]:
            combined_cs.append(cs_tpc[i])
        else:
            combined_cs.append(cs_nv[i])

    combined_chunk_start = np.array(combined_cs)
    combined_chunk_end = np.append(combined_chunk_start[1:], np.max([ce_tpc[-1], ce_nv[-1]]))
    
    
    return combined_chunk_start.astype(np.int64), combined_chunk_end.astype(np.int64)