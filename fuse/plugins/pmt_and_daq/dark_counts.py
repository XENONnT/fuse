import strax
import straxen
import logging
import numpy as np

from ...plugin import FuseBasePlugin
from ...common import pmt_gains, build_photon_propagation_output
from ...common import (
    init_spe_scaling_factor_distributions,
    photon_gain_calculation,
)

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.pmt_and_daq.dark_counts")


@export
class DarkCounts(FuseBasePlugin):
    """Plugin to simulate dark counts in a window around the physics
    interaction."""

    __version__ = "0.0.1"

    depends_on = "microphysics_summary"
    provides = "dark_count_photons"
    data_kind = "dark_count_photons"

    save_when = strax.SaveWhen.TARGET

    dtype = [
        (("PMT channel of the photon", "channel"), np.int16),
        (("Photon creates a double photo-electron emission", "dpe"), np.bool_),
        (("Sampled PMT gain for the photon", "photon_gain"), np.int32),
        (("ID of the cluster creating the photon", "cluster_id"), np.int32),
        (
            ("Type of the photon. S1 (1), S2 (2), PMT AP (0) or dark count (3)", "photon_type"),
            np.int8,
        ),
    ]
    dtype = dtype + strax.time_fields

    # Config options

    enable_dark_counts = straxen.URLConfig(
        default=False,
        type=bool,
        track=True,
        help="Decide if you want to to enable dark count simulation",
    )

    # Get this value from a database. For now lets just set it to 50 Hz per PMT
    dark_count_rate = straxen.URLConfig(
        default=50 * 494,
        type=(int, float),
        track=True,
        help="Rate of dark counts in Hz combined for all PMTs",
    )

    dark_count_left_window = straxen.URLConfig(
        default=2e6,
        type=int,
        track=True,
        help="Left window of the dark count simulation",
    )

    dark_count_right_window = straxen.URLConfig(
        default=2e6,
        type=int,
        track=True,
        help="Right window of the dark count simulation",
    )

    # Add a default pointing to the database
    dark_count_probability_per_pmt = straxen.URLConfig(
        track=True,
        help="Probability of dark counts per PMT",
    )

    # reconsider this one for dark counts....
    p_double_pe_emision = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=p_double_pe_emision",
        type=(int, float),
        cache=True,
        help="Probability of double photo-electron emission",
    )

    pmt_circuit_load_resistor = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=pmt_circuit_load_resistor",
        type=(int, float),
        cache=True,
        help="PMT circuit load resistor [kg m^2/(s^3 A)]",
    )

    digitizer_bits = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?&fmt=json&take=digitizer_bits",
        type=(int, float),
        cache=True,
        help="Number of bits of the digitizer boards",
    )

    digitizer_voltage_range = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=digitizer_voltage_range",
        type=(int, float),
        cache=True,
        help="Voltage range of the digitizer boards [V]",
    )

    gain_model_mc = straxen.URLConfig(
        default="cmt://to_pe_model?version=ONLINE&run_id=plugin.run_id",
        infer_type=False,
        help="PMT gain model",
    )

    photon_area_distribution = straxen.URLConfig(
        default="simple_load://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=photon_area_distribution"
        "&fmt=csv",
        cache=True,
        help="Photon area distribution",
    )

    n_tpc_pmts = straxen.URLConfig(
        type=int,
        help="Number of PMTs in the TPC",
    )

    def setup(self):
        super().setup()

        # self.dark_count_probability_per_pmt = np.ones(494) / 494 # uniform distribution

        self.gains = pmt_gains(
            self.gain_model_mc,
            digitizer_voltage_range=self.digitizer_voltage_range,
            digitizer_bits=self.digitizer_bits,
            pmt_circuit_load_resistor=self.pmt_circuit_load_resistor,
        )

        self.pmt_mask = np.array(self.gains) > 0  # Converted from to pe (from cmt by default)
        self.turned_off_pmts = np.nonzero(np.array(self.gains) == 0)[0]

        self.spe_scaling_factor_distributions = init_spe_scaling_factor_distributions(
            self.photon_area_distribution
        )

    def compute(self, interactions_in_roi, start, end):
        if not self.enable_dark_counts or (len(interactions_in_roi) == 0):
            return np.zeros(0, dtype=self.dtype)

        single_simulation_window = np.zeros(len(interactions_in_roi), dtype=strax.interval_dtype)
        single_simulation_window["time"] = interactions_in_roi["time"]
        single_simulation_window["dt"] = np.ones(len(interactions_in_roi))

        simulation_windows = strax.concat_overlapping_hits(
            single_simulation_window,
            extensions=(self.dark_count_left_window, self.dark_count_right_window),
            pmt_channels=(0, self.n_tpc_pmts),
            start=start,
            end=end,
        )

        # Get the number of dark counts in the simulation window
        expected_dark_counts_in_simulation_window = (
            simulation_windows["length"].astype(np.int64) * self.dark_count_rate / 1e9
        )
        dark_counts_in_simulation_window = self.rng.poisson(
            expected_dark_counts_in_simulation_window
        )

        dark_count_times = get_random_times(
            simulation_windows["length"], dark_counts_in_simulation_window, self.rng
        )
        dark_count_times += np.repeat(simulation_windows["time"], dark_counts_in_simulation_window)

        # distribute the dark counts to the PMTs
        dark_count_channels = self.rng.choice(
            self.n_tpc_pmts, len(dark_count_times), p=self.dark_count_probability_per_pmt
        )

        # We for sure need to update the inputs for this step
        photon_gains, photon_is_dpe = photon_gain_calculation(
            _photon_channels=dark_count_channels,
            p_double_pe_emision=self.p_double_pe_emision,
            gains=self.gains,
            spe_scaling_factor_distributions=self.spe_scaling_factor_distributions,
            rng=self.rng,
        )
        photon_is_dpe = False

        # now build the output
        result = build_photon_propagation_output(
            dtype=self.dtype,
            _photon_timings=dark_count_times,
            _photon_channels=dark_count_channels,
            _photon_gains=photon_gains,
            _photon_is_dpe=photon_is_dpe,
            _cluster_id=0,  # rethink this part...
            photon_type=3,
        )

        return result


# For sure this can be done more efficiently
def get_random_times(interval_length, number_of_entries, rng):
    times = []
    for max_time, n in zip(interval_length, number_of_entries):
        times.append(rng.uniform(0, max_time, n))
    return np.concatenate(times)
