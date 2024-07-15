import os
from typing import Tuple

import uproot
import awkward as ak
import numpy as np
import pandas as pd
import strax
import straxen

from ...dtypes import g4_fields, primary_positions_fields, deposit_positions_fields
from ...common import full_array_to_numpy, reshape_awkward, dynamic_chunking, awkward_to_flat_numpy
from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()


# Remove the path and file name option from the config and do this with the run_number??
@export
class ChunkInput(FuseBasePlugin):
    """Plugin to read XENONnT Geant4 root or csv files.

    The plugin can distribute the events in time based on a source rate
    and will create multiple chunks of data if needed.
    """

    __version__ = "0.3.4"

    depends_on: Tuple = tuple()
    provides = "geant4_interactions"

    dtype = deposit_positions_fields + g4_fields + primary_positions_fields + strax.time_fields

    save_when = strax.SaveWhen.TARGET

    source_done = False

    # Config options
    path = straxen.URLConfig(
        track=False,
        help="Path to the input file",
    )

    file_name = straxen.URLConfig(
        track=False,
        help="Name of the input file",
    )

    separation_scale = straxen.URLConfig(
        default=1e8,
        type=(int, float),
        help="Separation scale for the dynamic chunking in [ns]",
    )

    source_rate = straxen.URLConfig(
        default=1,
        type=(int, float),
        help="Source rate used to generate event times"
        "Use a value >0 to generate event times in fuse"
        "Use source_rate = 0 to use event times from the input file (only for csv input)",
    )

    fixed_event_spacing = straxen.URLConfig(
        default=False,
        type=bool,
        help="If True, the events will be spaced with a fixed time difference of 1/source_rate",
    )

    cut_delayed = straxen.URLConfig(
        default=9e18,
        type=(int, float),
        help="All interactions happening after this time (including the event time) will be cut",
    )

    n_interactions_per_chunk = straxen.URLConfig(
        default=1e5,
        type=(int, float),
        help="Minimum number of interaction per chunk",
    )

    entry_start = straxen.URLConfig(
        default=0,
        type=(int, float),
        help="Geant4 event to start simulation from",
    )

    entry_stop = straxen.URLConfig(
        default=None,
        help="Geant4 event to stop simulation at. If None, all events are simulated",
    )

    cut_by_eventid = straxen.URLConfig(
        default=False,
        type=bool,
        help="If selected, the next two arguments act on the G4 event id, "
        "and not the entry number (default)",
    )

    nr_only = straxen.URLConfig(
        default=False,
        type=bool,
        help="Filter only nuclear recoil events (maximum ER energy deposit 10 keV)",
    )

    def setup(self):
        super().setup()

        self.file_reader = file_loader(
            self.path,
            self.file_name,
            self.rng,
            separation_scale=self.separation_scale,
            event_rate=self.source_rate,
            cut_delayed=self.cut_delayed,
            n_interactions_per_chunk=self.n_interactions_per_chunk,
            arg_debug=self.debug,
            outer_cylinder=None,  # This is not running
            entry_start=self.entry_start,
            entry_stop=self.entry_stop,
            cut_by_eventid=self.cut_by_eventid,
            cut_nr_only=self.nr_only,
            fixed_event_spacing=self.fixed_event_spacing,
            log=self.log,
        )
        self.file_reader_iterator = self.file_reader.output_chunk()

    def compute(self):
        try:
            chunk_data, chunk_left, chunk_right, source_done = next(self.file_reader_iterator)
            chunk_data["endtime"] = chunk_data["time"]

            self.source_done = source_done

            return self.chunk(start=chunk_left, end=chunk_right, data=chunk_data)

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


class file_loader:
    """Load the complete root file and return interactions in chunks."""

    def __init__(
        self,
        directory,
        file_name,
        random_number_generator,
        separation_scale=1e8,
        event_rate=1,
        n_interactions_per_chunk=500,
        cut_delayed=4e12,
        last_chunk_length=1e8,
        first_chunk_left=1e6,
        chunk_delay_fraction=0.75,
        arg_debug=False,
        outer_cylinder=None,
        entry_start=None,
        entry_stop=None,
        cut_by_eventid=False,
        cut_nr_only=False,
        fixed_event_spacing=False,
        log=None,
    ):
        self.directory = directory
        self.file_name = file_name
        self.file_type = self.file_name.split(".")[-1]
        self.rng = random_number_generator
        self.separation_scale = separation_scale
        self.event_rate = event_rate / 1e9  # Conversion to ns
        self.n_interactions_per_chunk = n_interactions_per_chunk
        self.cut_delayed = cut_delayed
        self.last_chunk_length = np.int64(last_chunk_length)
        self.first_chunk_left = np.int64(first_chunk_left)
        self.chunk_delay_fraction = chunk_delay_fraction
        self.arg_debug = arg_debug
        self.outer_cylinder = outer_cylinder
        self.entry_start = entry_start
        self.entry_stop = entry_stop
        self.cut_by_eventid = cut_by_eventid
        self.cut_nr_only = cut_nr_only
        self.fixed_event_spacing = fixed_event_spacing
        self.log = log

        self.file = os.path.join(self.directory, self.file_name)

        self.dtype = deposit_positions_fields + g4_fields
        self.columns = list(np.dtype(self.dtype).names)
        # Remove eventid as it is not in the usual root or csv file
        self.columns.remove("eventid")
        self.dtype += primary_positions_fields + strax.time_fields

        # Prepare cut for root and csv case
        if self.outer_cylinder:
            self.cut_string = (
                f'(r < {self.outer_cylinder["max_r"]}) '
                f'& ((zp >= {self.outer_cylinder["min_z"] * 10}) '
                f'& (zp < {self.outer_cylinder["max_z"] * 10}))'
            )
        else:
            self.cut_string = None

    def output_chunk(self):
        """Function to return one chunk of data from the root or csv file."""

        if self.file_type == "root":
            interactions, n_simulated_events, start, stop = self._load_root_file()
        elif self.file_type == "csv":
            interactions, n_simulated_events, start, stop = self._load_csv_file()
        else:
            raise ValueError(
                f'Cannot load events from file "{self.file}": .root or .cvs file needed.'
            )

        # Removing all events with zero energy deposit
        # m = interactions["ed"] > 0

        if self.cut_nr_only:
            self.log.info("'nr_only' set to True, keeping only the NR events")
            m = ((interactions["type"] == "neutron") & (interactions["edproc"] == "hadElastic")) | (
                interactions["edproc"] == "ionIoni"
            )
            e_dep_er = ak.sum(interactions[~m]["ed"], axis=1)
            e_dep_nr = ak.sum(interactions[m]["ed"], axis=1)
            interactions = interactions[(e_dep_er < 10) & (e_dep_nr > 0)]

        # Removing all events with no interactions:
        m = ak.num(interactions["ed"]) > 0
        # and all events with no deposited energy
        m = m & (ak.sum(interactions["ed"], axis=1) > 0)

        interactions = interactions[m]

        # Sort interactions in events by time and subtract time of the first interaction
        interactions = interactions[ak.argsort(interactions["t"])]

        if self.event_rate > 0:
            interactions["t"] = interactions["t"] - interactions["t"][:, 0]

        # Adjust event times if necessary
        if self.event_rate > 0:
            num_interactions = len(interactions["t"])

            if self.fixed_event_spacing:
                self.log.info("Using fixed event spacing.")
                event_times = (
                    np.arange(
                        start=0, stop=num_interactions / self.event_rate, step=1 / self.event_rate
                    )
                    + 1e9
                )  # ns
            else:
                self.log.info("Using random event times.")
                event_times = self.rng.uniform(
                    low=start / self.event_rate, high=stop / self.event_rate, size=num_interactions
                ).astype(np.int64)
                event_times = np.sort(event_times)

            interactions["time"] = interactions["t"] + event_times

        elif self.event_rate == 0:
            self.log.info("Using event times from provided input file.")
            if self.file_type == "root":
                msg = (
                    "Using event times from root file is not recommended! "
                    "Use a source_rate > 0 instead."
                )
                self.log.warning(msg)
            interactions["time"] = interactions["t"]

        else:
            raise ValueError("Source rate cannot be negative!")

        interactions = interactions[interactions["t"] < self.cut_delayed]

        # Make into a flat numpy array
        interaction_time = awkward_to_flat_numpy(interactions["time"])

        # First caclulate sort index for the interaction times
        sort_idx = np.argsort(interaction_time)
        # and now make it an integer for strax time field
        interaction_time = interaction_time.astype(np.int64)
        # Sort the interaction times
        interaction_time = interaction_time[sort_idx]

        chunk_idx = dynamic_chunking(
            interaction_time, scale=self.separation_scale, n_min=self.n_interactions_per_chunk
        )

        unique_chunk_index_values = np.unique(chunk_idx)

        chunk_start = np.array(
            [interaction_time[chunk_idx == i][0] for i in unique_chunk_index_values]
        )
        chunk_end = np.array(
            [interaction_time[chunk_idx == i][-1] for i in unique_chunk_index_values]
        )

        if (len(chunk_start) > 1) & (len(chunk_end) > 1):
            gap_length = chunk_start[1:] - chunk_end[:-1]
            gap_length = np.append(gap_length, gap_length[-1] + self.last_chunk_length)
            chunk_bounds = chunk_end + np.int64(self.chunk_delay_fraction * gap_length)
            self.chunk_bounds = np.append(chunk_start[0] - self.first_chunk_left, chunk_bounds)

        else:
            self.log.warning(
                "Only one Chunk created! Only a few events simulated? "
                "If no, your chunking parameters might not be optimal. "
                "Try to decrease the source_rate or decrease the n_interactions_per_chunk."
            )
            self.chunk_bounds = [
                chunk_start[0] - self.first_chunk_left,
                chunk_end[0] + self.last_chunk_length,
            ]

        # We need to get the min and max times for each event
        # to preselect events with interactions in the chunk bounds
        times_min = ak.to_numpy(ak.min(interactions["time"], axis=1)).astype(np.int64)
        times_max = ak.to_numpy(ak.max(interactions["time"], axis=1)).astype(np.int64)

        # Process and yield each chunk
        source_done = False
        self.log.info(f"Simulating data in {len(unique_chunk_index_values)} chunks.")
        for c_ix, chunk_left, chunk_right in zip(
            unique_chunk_index_values, self.chunk_bounds[:-1], self.chunk_bounds[1:]
        ):

            # We do a preselection of the events that have interactions within the chunk
            # before converting the full array to numpy (which is expensive in terms of memory)
            m = (times_min <= chunk_right) & (times_max >= chunk_left)
            current_chunk = interactions[m]

            if len(current_chunk) == 0:
                current_chunk = np.empty(0, dtype=self.dtype)

            else:
                # Convert the chunk from awkward array to a numpy array
                current_chunk = full_array_to_numpy(current_chunk, self.dtype)

            # Now we have the chunk of data in strax/numpy format
            # We can now filter only the interactions within the chunk
            select_times = current_chunk["time"] >= chunk_left
            select_times &= current_chunk["time"] <= chunk_right
            current_chunk = current_chunk[select_times]

            # Sorting each chunk by time within the chunk
            sort_chunk = np.argsort(current_chunk["time"])
            current_chunk = current_chunk[sort_chunk]

            if c_ix == unique_chunk_index_values[-1]:
                source_done = True

            yield current_chunk, chunk_left, chunk_right, source_done

    def last_chunk_bounds(self):
        return self.chunk_bounds[-1]

    def _load_root_file(self):
        """Function which reads a root file using uproot, performs a simple cut
        and builds an awkward array.

        Returns:
            interactions: awkward array
            n_simulated_events: Total number of simulated events
            start: Index of the first loaded interaction
            stop: Index of the last loaded interaction
        """
        ttree, n_simulated_events = self._get_ttree()

        if self.arg_debug:
            self.log.info(f"Total entries in input file = {ttree.num_entries}")
            cutby_string = "output file entry"
            if self.cut_by_eventid:
                cutby_string = "g4 eventid"

            if self.entry_start is not None:
                self.log.debug(f"Starting to read from {cutby_string} {self.entry_start}")
            if self.entry_stop is not None:
                self.log.debug(f"Ending read in at {cutby_string} {self.entry_stop}")

        if self.entry_start is not None and self.entry_stop is not None:
            if self.entry_start >= self.entry_stop:
                raise ValueError(
                    "The requested range is not valid! "
                    "Make sure that entry_stop is larger than entry_start"
                )

        # If we cut by eventid we have to read all of them first to find the start and stop index
        if self.cut_by_eventid:
            all_eventids = ttree.arrays("eventid")

            if self.entry_start is not None:
                if self.entry_start > np.max(all_eventids["eventid"]):
                    raise ValueError(
                        "The requested eventid range is not in the file! "
                        "Maybe you want to set cut_by_eventid to False?"
                    )
                start_index = np.searchsorted(all_eventids["eventid"], self.entry_start)
            else:
                start_index = 0

            if self.entry_stop is not None:
                if self.entry_stop <= np.min(all_eventids["eventid"]):
                    raise ValueError(
                        "The requested eventid range is not in the file! "
                        "Maybe you want to set cut_by_eventid to False?"
                    )
                stop_index = np.searchsorted(all_eventids["eventid"], self.entry_stop)
            else:
                stop_index = n_simulated_events

        else:
            entries = len(ttree.arrays("eventid"))
            if self.entry_start is not None:
                if self.entry_start > entries:
                    raise ValueError("The requested entry range is not in the file!")
                start_index = max(0, self.entry_start)
            else:
                start_index = 0

            if self.entry_stop is not None:
                if self.entry_stop < 0:
                    raise ValueError("The requested entry range is not in the file!")
                stop_index = min(self.entry_stop, entries)
            else:
                stop_index = entries

        n_simulated_events = stop_index - start_index
        if n_simulated_events <= 0:
            raise ValueError(
                "No events selected! Check entry_start, entry_stop and cut_by_eventid."
            )

        # Conversions and parameters to be computed:
        alias = {
            "x": "xp/10",  # converting "geant4" mm to "straxen" cm
            "y": "yp/10",
            "z": "zp/10",
            "r": "sqrt(x**2 + y**2)",
            "t": "time*10**9",
        }

        # Read in data, convert mm to cm and perform a first cut if specified:
        interactions = ttree.arrays(
            self.columns,
            self.cut_string,
            aliases=alias,
            entry_start=start_index,
            entry_stop=stop_index,
        )
        eventids = ttree.arrays("eventid", entry_start=start_index, entry_stop=stop_index)
        eventids = ak.broadcast_arrays(eventids["eventid"], interactions["x"])[0]
        interactions["eventid"] = eventids

        xyz_pri = ttree.arrays(
            ["x_pri", "y_pri", "z_pri"],
            aliases={"x_pri": "xp_pri/10", "y_pri": "yp_pri/10", "z_pri": "zp_pri/10"},
            entry_start=start_index,
            entry_stop=stop_index,
        )

        interactions["x_pri"] = ak.broadcast_arrays(xyz_pri["x_pri"], interactions["x"])[0]
        interactions["y_pri"] = ak.broadcast_arrays(xyz_pri["y_pri"], interactions["x"])[0]
        interactions["z_pri"] = ak.broadcast_arrays(xyz_pri["z_pri"], interactions["x"])[0]

        return interactions, n_simulated_events, start_index, stop_index

    def _get_ttree(self):
        """Function which searches for the correct ttree in MC root file.

        Args:
            directory: Directory where file is
            file_name: Name of the file
        Returns:
            root ttree and number of simulated events
        """
        root_dir = uproot.open(self.file)

        # Searching for TTree according to old/new MC file structure:
        if root_dir.classname_of("events") == "TTree":
            ttree = root_dir["events"]
            n_simulated_events = root_dir["nEVENTS"].members["fVal"]
        elif root_dir.classname_of("events/events") == "TTree":
            ttree = root_dir["events/events"]
            n_simulated_events = root_dir["events/nbevents"].members["fVal"]
        else:
            ttrees = []
            for k, v in root_dir.classnames().items():
                if v == "TTree":
                    ttrees.append(k)
            raise ValueError(
                f'Cannot find ttree object of "{self.file}".'
                "I tried to search in events and events/events."
                f"Found a ttree in {ttrees}?"
            )
        return ttree, n_simulated_events

    def _load_csv_file(self):
        """Function which reads a csv file using pandas, performs a simple cut
        and builds an awkward array.

        Returns:
            interactions: awkward array
            n_simulated_events: Total number of simulated events
            start: Index of the first loaded interaction
            stop: Index of the last loaded interaction
        """

        self.log.debug("Load instructions from a csv file!")

        df = pd.read_csv(self.file)

        # unit conversion similar to root case
        df["x"] = df["xp"] / 10
        df["y"] = df["yp"] / 10
        df["z"] = df["zp"] / 10
        df["x_pri"] = df["xp_pri"] / 10
        df["y_pri"] = df["yp_pri"] / 10
        df["z_pri"] = df["zp_pri"] / 10
        df["r"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
        df["t"] = df["time"]

        missing_columns = set(self.columns) - set(df.columns)

        # Check if all needed columns are in place:
        if missing_columns:
            raise ValueError(f"Not all needed columns provided! {missing_columns} are missing.")

        n_simulated_events = len(np.unique(df.eventid))

        if self.outer_cylinder:
            df = df.query(self.cut_string)

        df = df[self.columns + ["eventid", "x_pri", "y_pri", "z_pri"]]

        interactions = self._awkwardify_df(df)

        # Use always all events in the csv file
        start = 0
        stop = n_simulated_events

        return interactions, n_simulated_events, start, stop

    @staticmethod
    def _awkwardify_df(df):
        """Function which builds an jagged awkward array from pandas dataframe.

        Args:
            df: Pandas Dataframe

        Returns:
            ak.Array(dictionary): awkward array
        """

        _, evt_offsets = np.unique(df["eventid"], return_counts=True)

        dictionary = {
            "x": reshape_awkward(df["x"].values, evt_offsets),
            "y": reshape_awkward(df["y"].values, evt_offsets),
            "z": reshape_awkward(df["z"].values, evt_offsets),
            "x_pri": reshape_awkward(df["x_pri"].values, evt_offsets),
            "y_pri": reshape_awkward(df["y_pri"].values, evt_offsets),
            "z_pri": reshape_awkward(df["z_pri"].values, evt_offsets),
            "t": reshape_awkward(df["t"].values, evt_offsets),
            "ed": reshape_awkward(df["ed"].values, evt_offsets),
            "type": reshape_awkward(np.array(df["type"], dtype=str), evt_offsets),
            "trackid": reshape_awkward(df["trackid"].values, evt_offsets),
            "parenttype": reshape_awkward(np.array(df["parenttype"], dtype=str), evt_offsets),
            "parentid": reshape_awkward(df["parentid"].values, evt_offsets),
            "creaproc": reshape_awkward(np.array(df["creaproc"], dtype=str), evt_offsets),
            "edproc": reshape_awkward(np.array(df["edproc"], dtype=str), evt_offsets),
            "eventid": reshape_awkward(df["eventid"].values, evt_offsets),
        }

        return ak.Array(dictionary)
