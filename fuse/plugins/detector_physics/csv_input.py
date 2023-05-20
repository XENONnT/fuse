import strax
import straxen
import os
import numba
import logging

import pandas as pd
import numpy as np

from ...common import dynamic_chunking

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.detector_physics.csv_input')
log.setLevel('WARNING')

@export
@strax.takes_config(
    strax.Option('input_file', track=False, infer_type=False,
                 help="CSV file to read"),
)
class ChunkCsvInput(strax.Plugin):
    """
    Plugin which reads a CSV file containing instructions for the detector physics simulation
    and returns the data in chunks
    """


    __version__ = "0.0.0"

    depends_on = tuple()
    provides = "microphysics_summary"
    data_kind = "clustered_interactions"

    #Forbid rechunking
    rechunk_on_save = False

    source_done = False

    dtype = [('x', np.float32),
             ('y', np.float32),
             ('z', np.float32),
             ('photons', np.int32),
             ('electrons', np.int32),
             ('excitons', np.int32),
             ('e_field', np.float32),
             ('ed', np.float32),
             ('nestid', np.int32),
             ('t', np.int32), #Remove them later as they are not in the usual micropyhsics summary
             ('eventid', np.int32),#Remove them later as they are not in the usual micropyhsics summary
            ]
    dtype = dtype + strax.time_fields

    #Config options
    debug = straxen.URLConfig(
        default=False, type=bool,
        help='Show debug informations',
    )

    separation_scale = straxen.URLConfig(
        default=1e8, type=(int, float),
        help='separation_scale',
    )

    source_rate = straxen.URLConfig(
        default=1, type=(int, float),
        help='source_rate',
    )

    n_interactions_per_chunk = straxen.URLConfig(
        default=25, type=(int, float),
        help='n_interactions_per_chunk',
    )

    def setup(self):

        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running ChunkCsvInput in debug mode")


        self.file_reader = csv_file_loader(
            input_file = self.input_file,
            event_rate = self.source_rate,
            separation_scale = self.separation_scale,
            n_interactions_per_chunk = self.n_interactions_per_chunk,
            debug = self.debug,
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
    

class csv_file_loader():
    """Class to load a CSV file with detector simulation instructions"""

    def __init__(self,
                 input_file,
                 event_rate,
                 separation_scale,
                 n_interactions_per_chunk,
                 chunk_delay_fraction = 0.75,
                 first_chunk_left = 1e6,
                 last_chunk_length = 1e8,
                 debug = False,
                 ):
        
        self.input_file = input_file
        self.event_rate = event_rate/ 1e9 #Conversion to ns 
        self.separation_scale = separation_scale
        self.n_interactions_per_chunk = n_interactions_per_chunk
        self.chunk_delay_fraction = chunk_delay_fraction
        self.last_chunk_length = np.int64(last_chunk_length)
        self.first_chunk_left = np.int64(first_chunk_left)
        self.debug = debug

        self.dtype = [('x', np.float32),
                      ('y', np.float32),
                      ('z', np.float32),
                      ('photons', np.int32),
                      ('electrons', np.int32),
                      ('excitons', np.int32),
                      ('e_field', np.float32),
                      ('ed', np.float32),
                      ('nestid', np.int32),
                      ('t', np.int32), #Remove them later as they are not in the usual micropyhsics summary
                      ('eventid', np.int32),#Remove them later as they are not in the usual micropyhsics summary
                      ]
        self.dtype = self.dtype + strax.time_fields

        #the csv file needs to have these columns:
        self.columns = ["x", "y", "z",
                        "photons", "electrons", "excitons",
                        "e_field", "ed", "nestid", "t", "eventid"]


    def output_chunk(self):
        
        instructions, n_simulated_events = self.__load_csv_file()

        #Assign event times and dynamic chunking
        event_times = np.random.uniform(low = 0,
                                        high = n_simulated_events/self.event_rate,
                                        size = n_simulated_events
                                        ).astype(np.int64)
        event_times = np.sort(event_times)

        structure = np.unique(instructions["eventid"], return_counts = True)[1]
        interaction_time = np.repeat(event_times[:len(structure)], structure)
        instructions["time"] = interaction_time + instructions["t"]

        sort_idx = np.argsort(instructions["time"])
        instructions = instructions[sort_idx]

        #Group into chunks
        chunk_idx = dynamic_chunking(instructions["time"],
                                     scale = self.separation_scale,
                                     n_min = self.n_interactions_per_chunk)
        
        #Calculate chunk start and end times
        chunk_start = np.array([instructions[chunk_idx == i][0]["time"] for i in np.unique(chunk_idx)])
        chunk_end = np.array([instructions[chunk_idx == i][-1]["time"] for i in np.unique(chunk_idx)])
        
        if (len(chunk_start) > 1) & (len(chunk_end) > 1):
            gap_length = chunk_start[1:] - chunk_end[:-1]
            gap_length = np.append(gap_length, gap_length[-1] + self.last_chunk_length)
            chunk_bounds = chunk_end + np.int64(self.chunk_delay_fraction*gap_length)
            self.chunk_bounds = np.append(chunk_start[0]-self.first_chunk_left, chunk_bounds)
        else: 
            log.warn("Only one Chunk! Rate to high?")
            self.chunk_bounds = [chunk_start[0] - self.first_chunk_left, chunk_end[0]+self.last_chunk_length]
        
        for c_ix, chunk_left, chunk_right in zip(np.unique(chunk_idx), self.chunk_bounds[:-1], self.chunk_bounds[1:]):
            
            yield instructions[chunk_idx == c_ix], chunk_left, chunk_right

    def last_chunk_bounds(self):
        return self.chunk_bounds[-1]
    
    def __load_csv_file(self):
        log.debug("Load detector simulation instructions from a csv file!")
        df = pd.read_csv(self.input_file)

        #Check if all needed columns are in place:
        if not set(self.columns).issubset(df.columns):
            log.warn("Not all needed columns provided!")

        n_simulated_events = len(np.unique(df.eventid))

        instructions = np.zeros(len(df), dtype=self.dtype)
        for column in df.columns:
            instructions[column] = df[column]

        return instructions, n_simulated_events