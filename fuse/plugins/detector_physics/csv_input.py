from typing import Tuple

import numpy as np
import pandas as pd
import strax
import straxen

from ...dtypes import (
    cluster_positions_fields,
    cluster_id_fields,
    csv_cluster_misc_fields,
    quanta_fields,
    electric_fields,
)
from ...common import dynamic_chunking
from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()
__all__.extend(["microphysics_summary_fields"])


# In some cases we might want to change dtype of microphysics_summary
# through microphysics_summary_fields, so do not set microphysics_summary_fields
# or variable related to it as a static attribute of a class.
microphysics_summary_fields = (
    cluster_positions_fields + quanta_fields + electric_fields + cluster_id_fields
)


@export
class ChunkCsvInput(FuseBasePlugin):
    """Plugin which reads a CSV file containing instructions for the detector
    physics simulation and returns the data in chunks."""

    __version__ = "0.2.2"

    depends_on: Tuple = tuple()
    provides = "microphysics_summary"
    data_kind = "interactions_in_roi"

    save_when = strax.SaveWhen.TARGET

    source_done = False

    # Config options
    input_file = straxen.URLConfig(
        track=False,
        infer_type=False,
        help="CSV file to read",
    )

    separation_scale = straxen.URLConfig(
        default=1e8,
        type=(int, float),
        help="Start a new chunk when the previous cluster is separated by this time scale",
    )

    source_rate = straxen.URLConfig(
        default=1,
        type=(int, float),
        help="Source rate used to generate event times. "
        "Use a value >0 to generate event times in fuse. "
        "Use source_rate = 0 to use event times from the input file (only for csv input).",
    )

    n_interactions_per_chunk = straxen.URLConfig(
        default=1e5,
        type=(int, float),
        help="n_interactions_per_chunk",
    )

    first_chunk_left = straxen.URLConfig(
        default=1e6,
        type=(int, float),
        help="Time left of the first chunk",
    )

    last_chunk_length = straxen.URLConfig(
        default=1e8,
        type=(int, float),
        help="Time length of the last chunk",
    )

    def infer_dtype(self):
        return microphysics_summary_fields + strax.time_fields

    @staticmethod
    def needed_csv_input_fields():
        return microphysics_summary_fields + csv_cluster_misc_fields

    def setup(self):
        super().setup()

        self.file_reader = csv_file_loader(
            input_file=self.input_file,
            random_number_generator=self.rng,
            event_rate=self.source_rate,
            separation_scale=self.separation_scale,
            n_interactions_per_chunk=self.n_interactions_per_chunk,
            first_chunk_left=self.first_chunk_left,
            last_chunk_length=self.last_chunk_length,
            debug=self.debug,
            log=self.log,
        )
        self.file_reader_iterator = self.file_reader.output_chunk()

    def compute(self):
        try:
            chunk_data, chunk_left, chunk_right, source_done = next(self.file_reader_iterator)
            chunk_data["endtime"] = chunk_data["time"]
            data = np.zeros(len(chunk_data), dtype=self.dtype)
            strax.copy_to_buffer(chunk_data, data, "_bring_data_into_correct_format")

            self.source_done = source_done

            return self.chunk(start=chunk_left, end=chunk_right, data=data)

        except StopIteration:
            raise RuntimeError("Bug in chunk building!")

    def source_finished(self):
        return self.source_done

    def is_ready(self, chunk_i):
        """Overwritten to mimic online input plugin.

        Returns False to check source finished; Returns True to get next
        chunk.
        """
        if "ready" not in self.__dict__:
            self.ready = False
        self.ready ^= True  # Flip
        return self.ready


class csv_file_loader:
    """Class to load a CSV file with detector simulation instructions."""

    def __init__(
        self,
        input_file,
        random_number_generator,
        event_rate,
        separation_scale,
        n_interactions_per_chunk,
        chunk_delay_fraction=0.75,
        first_chunk_left=1e6,
        last_chunk_length=1e8,
        debug=False,
        log=None,
    ):
        self.input_file = input_file
        self.rng = random_number_generator
        self.event_rate = event_rate / 1e9  # Conversion to ns
        self.separation_scale = separation_scale
        self.n_interactions_per_chunk = n_interactions_per_chunk
        self.chunk_delay_fraction = chunk_delay_fraction
        self.last_chunk_length = np.int64(last_chunk_length)
        self.first_chunk_left = np.int64(first_chunk_left)
        self.debug = debug
        self.log = log

        # The csv file needs to have these columns:
        _fields = ChunkCsvInput.needed_csv_input_fields()
        self.columns = list(np.dtype(_fields).names)
        self.dtype = _fields + strax.time_fields

    def output_chunk(self):
        instructions, n_simulated_events = self.__load_csv_file()

        # Assign event times and dynamic chunking
        if self.event_rate > 0:
            event_times = self.rng.uniform(
                low=0, high=n_simulated_events / self.event_rate, size=n_simulated_events
            ).astype(np.int64)
            event_times = np.sort(event_times)

            structure = np.unique(instructions["eventid"], return_counts=True)[1]
            interaction_time = np.repeat(event_times[: len(structure)], structure)
            instructions["time"] = interaction_time + instructions["t"]
        elif self.event_rate == 0:
            instructions["time"] = instructions["t"]
            self.log.debug("Using event times from provided input file.")
        else:
            raise ValueError("Source rate cannot be negative!")

        sort_idx = np.argsort(instructions["time"])
        instructions = instructions[sort_idx]

        # Group into chunks
        chunk_idx = dynamic_chunking(
            instructions["time"], scale=self.separation_scale, n_min=self.n_interactions_per_chunk
        )

        # Calculate chunk start and end times
        chunk_start = np.array(
            [instructions[chunk_idx == i][0]["time"] for i in np.unique(chunk_idx)]
        )
        chunk_end = np.array(
            [instructions[chunk_idx == i][-1]["time"] for i in np.unique(chunk_idx)]
        )

        if (len(chunk_start) > 1) & (len(chunk_end) > 1):
            gap_length = chunk_start[1:] - chunk_end[:-1]
            gap_length = np.append(gap_length, gap_length[-1] + self.last_chunk_length)
            chunk_bounds = chunk_end + np.int64(self.chunk_delay_fraction * gap_length)
            self.chunk_bounds = np.append(chunk_start[0] - self.first_chunk_left, chunk_bounds)
        else:
            self.log.warning("Only one Chunk! Rate too high?")
            self.chunk_bounds = [
                chunk_start[0] - self.first_chunk_left,
                chunk_end[0] + self.last_chunk_length,
            ]

        source_done = False
        unique_chunk_index_values = np.unique(chunk_idx)
        for c_ix, chunk_left, chunk_right in zip(
            unique_chunk_index_values, self.chunk_bounds[:-1], self.chunk_bounds[1:]
        ):
            if c_ix == unique_chunk_index_values[-1]:
                source_done = True
                self.log.debug("Build last chunk.")

            yield instructions[chunk_idx == c_ix], chunk_left, chunk_right, source_done

    def last_chunk_bounds(self):
        return self.chunk_bounds[-1]

    def __load_csv_file(self):
        self.log.debug("Load detector simulation instructions from a csv file!")
        df = pd.read_csv(self.input_file)

        missing_columns = set(self.columns) - set(df.columns)

        # Check if all needed columns are in place:
        if missing_columns:
            raise ValueError(f"Not all needed columns provided! {missing_columns} are missing.")

        n_simulated_events = len(np.unique(df.eventid))

        instructions = np.zeros(len(df), dtype=self.dtype)
        for column in df.columns:
            instructions[column] = df[column]

        return instructions, n_simulated_events
