import logging

import numpy as np
import nestpy
import strax
import straxen

from ...common import pmt_gains, build_photon_propagation_output
from ...common import (
    init_spe_scaling_factor_distributions,
    pmt_transit_time_spread,
    photon_gain_calculation,
)
from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.detector_physics.s1_photon_propagation")

# Initialize the nestpy random generator
# The seed will be set in the compute method
nest_rng = nestpy.RandomGen.rndm()


@export
class S1PhotonPropagationBase(FuseBasePlugin):
    """Base plugin to simulate the propagation of S1 photons in the detector.
    Photons are randomly assigned to PMT channels based on their starting
    position and the timing of the photons is calculated.

    Note: The timing calculation is defined in the child plugin.
    """

    __version__ = "0.3.0"

    depends_on = ("s1_photons", "microphysics_summary")
    provides = "propagated_s1_photons"
    data_kind = "S1_photons"

    save_when = strax.SaveWhen.TARGET

    dtype = [
        (("PMT channel of the photon", "channel"), np.int16),
        (("Photon creates a double photo-electron emission", "dpe"), np.bool_),
        (("Sampled PMT gain for the photon", "photon_gain"), np.int32),
        (("ID of the cluster creating the photon", "cluster_id"), np.int32),
        (("Type of the photon. S1 (1), S2 (2) or PMT AP (0)", "photon_type"), np.int8),
    ]
    dtype = dtype + strax.time_fields

    # Config options shared by S1 and S2 simulation
    p_double_pe_emision = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=p_double_pe_emision",
        type=(int, float),
        cache=True,
        help="Probability of double photo-electron emission",
    )

    pmt_transit_time_spread = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=pmt_transit_time_spread",
        type=(int, float),
        cache=True,
        help="Spread of the PMT transit times [ns]",
    )

    pmt_transit_time_mean = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=pmt_transit_time_mean",
        type=(int, float),
        cache=True,
        help="Mean of the PMT transit times [ns]",
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

    n_top_pmts = straxen.URLConfig(
        type=int,
        help="Number of PMTs on top array",
    )

    n_tpc_pmts = straxen.URLConfig(
        type=int,
        help="Number of PMTs in the TPC",
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

    s1_pattern_map = straxen.URLConfig(
        default="pattern_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=s1_pattern_map"
        "&fmt=pkl"
        "&pmt_mask=plugin.pmt_mask",
        cache=True,
        help="S1 pattern map",
    )

    def setup(self):
        super().setup()

        if self.deterministic_seed:
            # Dont know but nestpy seems to have a problem with large seeds
            self.short_seed = int(repr(self.seed)[-8:])
            log.debug(f"Generating nestpy random numbers from seed {self.short_seed}")
        else:
            log.debug("Generating random numbers with seed pulled from OS")

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

    def compute(self, interactions_in_roi):
        # Just apply this to clusters with photons hitting a PMT
        instruction = interactions_in_roi[interactions_in_roi["n_s1_photon_hits"] > 0]

        if len(instruction) == 0:
            return np.zeros(0, self.dtype)

        # set the global nest random generator with self.short_seed
        nest_rng.set_seed(self.short_seed)
        # Now lock the seed during the computation
        nest_rng.lock_seed()
        # increment the seed. Next chunk we will use the modified seed to generate random numbers
        self.short_seed += 1

        t = instruction["time"]
        x = instruction["x"]
        y = instruction["y"]
        z = instruction["z"]
        n_photons = instruction["photons"].astype(np.int64)
        recoil_type = instruction["nestid"]
        positions = np.array([x, y, z]).T  # For map interpolation

        _cluster_id = np.repeat(instruction["cluster_id"], instruction["n_s1_photon_hits"])

        # The new way interpolation is written always require a list
        _photon_channels = self.photon_channels(
            positions=positions,
            n_photon_hits=instruction["n_s1_photon_hits"],
        )

        _photon_timings = self.photon_timings(
            t=t,
            n_photon_hits=instruction["n_s1_photon_hits"],
            recoil_type=recoil_type,
            channels=_photon_channels,
            positions=positions,
            e_dep=instruction["ed"],
            n_photons_emitted=n_photons,
            n_excitons=instruction["excitons"].astype(np.int64),
            local_field=instruction["e_field"],
        )

        # I should sort by time i guess
        sortind = np.argsort(_photon_timings)
        _photon_channels = _photon_channels[sortind]
        _photon_timings = _photon_timings[sortind]
        _cluster_id = _cluster_id[sortind]

        # Do i want to save both -> timings with and without pmt transit time spread?
        # Correct for PMT transit Time Spread

        _photon_timings = pmt_transit_time_spread(
            _photon_timings=_photon_timings,
            pmt_transit_time_mean=self.pmt_transit_time_mean,
            pmt_transit_time_spread=self.pmt_transit_time_spread,
            rng=self.rng,
        )

        _photon_gains, _photon_is_dpe = photon_gain_calculation(
            _photon_channels=_photon_channels,
            p_double_pe_emision=self.p_double_pe_emision,
            gains=self.gains,
            spe_scaling_factor_distributions=self.spe_scaling_factor_distributions,
            rng=self.rng,
        )

        result = build_photon_propagation_output(
            dtype=self.dtype,
            _photon_timings=_photon_timings,
            _photon_channels=_photon_channels,
            _photon_gains=_photon_gains,
            _photon_is_dpe=_photon_is_dpe,
            _cluster_id=_cluster_id,
            photon_type=1,
        )

        result = strax.sort_by_time(result)

        # Unlock the nest random generator seed again
        nest_rng.unlock_seed()

        return result

    def photon_channels(self, positions, n_photon_hits):
        """Calculate photon arrival channels :params positions: 2d array with
        xy positions of interactions :params n_photon_hits: 1d array of ints
        with number of photon hits to simulate :params config: dict wfsim
        config :params s1_pattern_map: interpolator instance of the s1 pattern
        map returns nested array with photon channels."""
        channels = np.arange(self.n_tpc_pmts)  # +1 for the channel map
        p_per_channel = self.s1_pattern_map(positions)
        p_per_channel[:, np.in1d(channels, self.turned_off_pmts)] = 0

        _photon_channels = []
        for ppc, n in zip(p_per_channel, n_photon_hits):
            _photon_channels.append(
                self.rng.choice(channels, size=n, p=ppc / np.sum(ppc), replace=True)
            )

        return np.concatenate(_photon_channels)

    def photon_timings(self):
        raise NotImplementedError  # To be implemented by child class


@export
class S1PhotonPropagation(S1PhotonPropagationBase):
    """Child plugin to simulate the propagation of S1 photons using optical
    propagation and luminescence timing from nestpy."""

    __version__ = "0.2.0"

    child_plugin = True

    maximum_recombination_time = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=maximum_recombination_time",
        type=(int, float),
        cache=True,
        help="Maximum recombination time [ns]",
    )

    s1_optical_propagation_spline = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=s1_time_spline"
        "&fmt=json.gz"
        "&method=RegularGridInterpolator",
        cache=True,
        help="Spline for the optical propagation of S1 signals",
    )

    def setup(self):
        super().setup()

        log.debug(
            "Using NEST for scintillation time without set calculator\n"
            "Creating new nestpy calculator"
        )
        self.nestpy_calc = nestpy.NESTcalc(nestpy.DetectorExample_XENON10())

    def photon_timings(
        self,
        t,
        n_photon_hits,
        recoil_type,
        channels,
        positions,
        e_dep,
        n_photons_emitted,
        n_excitons,
        local_field,
    ):
        """Calculate distribution of photon arrival timnigs
        Args:
            t: 1d array of ints
            n_photon_hits: number of photon hits, 1d array of ints
            recoil_type: 1d array of ints
            config: dict wfsim config
            channels: list of photon hit channels
            positions: nx3 array of true XYZ positions from instruction
            e_dep: energy of the deposit, 1d float array
            n_photons_emitted: number of orignally emitted photons/quanta, 1d int array
            n_excitons: number of exctions in deposit, 1d int array
            local_field: local field in the point of the deposit, 1d array of floats
        Returns:
            photon timing array
        """
        _photon_timings = np.repeat(t, n_photon_hits)

        z_positions = np.repeat(positions[:, 2], n_photon_hits)

        # Propagation Modeling
        _photon_timings += self.optical_propagation(
            channels,
            z_positions,
        ).astype(np.int64)

        # Scintillation Modeling
        counts_start = 0
        for i, counts in enumerate(n_photon_hits):
            # Allow overwriting with "override_s1_photon_time_field"
            # xenon:j_angevaare:wfsim_photon_timing_bug
            # _local_field = config.get('override_s1_photon_time_field', local_field[i])
            # _local_field = (_local_field if _local_field >0 else local_field[i])
            _local_field = local_field[i]
            scint_time = self.nestpy_calc.GetPhotonTimes(
                nestpy.INTERACTION_TYPE(recoil_type[i]),
                n_photons_emitted[i],
                n_excitons[i],
                _local_field,
                e_dep[i],
            )

            scint_time = np.clip(scint_time, 0, self.maximum_recombination_time)

            # The first part of the scint_time is from exciton only, see
            # https://github.com/NESTCollaboration/nestpy/blob/fe3d5d7da5d9b33ac56fbea519e02ef55152bc1d/src/nestpy/NEST.cpp#L164-L179
            _photon_timings[counts_start : counts_start + counts] += self.rng.choice(
                scint_time, counts, replace=False
            ).astype(np.int64)

            counts_start += counts

        return _photon_timings

    def optical_propagation(self, channels, z_positions):
        """Function gettting times from s1 timing splines:
        Args:
            channels: The channels of all s1 photon
            z_positions: The Z positions of all s1 photon
        """
        assert len(z_positions) == len(channels), "Give each photon a z position"

        prop_time = np.zeros_like(channels)
        z_rand = np.array([z_positions, self.rng.random(len(channels))]).T

        is_top = channels < self.n_top_pmts
        prop_time[is_top] = self.s1_optical_propagation_spline(z_rand[is_top], map_name="top")

        is_bottom = channels >= self.n_top_pmts
        prop_time[is_bottom] = self.s1_optical_propagation_spline(
            z_rand[is_bottom], map_name="bottom"
        )

        return prop_time
