import strax
import straxen
import numpy as np

from numba import njit
from scipy.stats import skewnorm
from scipy import constants

from ...dtypes import propagated_photons_fields
from ...common import pmt_gains, build_photon_propagation_output
from ...common import (
    init_spe_scaling_factor_distributions,
    pmt_transit_time_spread,
    photon_gain_calculation,
)
from ...plugin import FuseBaseDownChunkingPlugin

export, __all__ = strax.exporter()

conversion_to_bar = 1 / constants.elementary_charge / 1e1


@export
class S2PhotonPropagationBase(FuseBaseDownChunkingPlugin):
    """Base plugin to simulate the propagation of S2 photons in the detector.
    Photons are randomly assigned to PMT channels based on their starting
    position and the timing of the photons is calculated.

    Note: The timing calculation is defined in the child plugin.
    """

    __version__ = "0.3.5"

    depends_on = (
        "merged_electron_time",
        "merged_s2_photons",
        "merged_extracted_electrons",
        "merged_drifted_electrons",
        "merged_s2_photons_sum",
        "merged_microphysics_summary",
    )

    provides = "propagated_s2_photons"
    data_kind = "s2_photons"

    save_when = strax.SaveWhen.TARGET

    dtype = propagated_photons_fields + strax.time_fields

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

    # Config options specific to S2 simulation
    phase_s2 = straxen.URLConfig(
        default="gas",
        help="Phase of the S2 producing region",
    )

    tpc_length = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?&fmt=json&take=tpc_length",
        type=(int, float),
        cache=True,
        help="Length of the XENONnT TPC [cm]",
    )

    tpc_radius = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?&fmt=json&take=tpc_radius",
        type=(int, float),
        cache=True,
        help="Radius of the XENONnT TPC [cm]",
    )

    diffusion_constant_transverse = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=diffusion_constant_transverse",
        type=(int, float),
        cache=True,
        help="Transverse diffusion constant [cm^2/ns]",
    )

    s2_aft_skewness = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=s2_aft_skewness",
        type=(int, float),
        cache=True,
        help="Skew of the S2 area fraction top",
    )

    s2_aft_sigma = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?&fmt=json&take=s2_aft_sigma",
        type=(int, float),
        cache=True,
        help="Width of the S2 area fraction top",
    )

    enable_diffusion_transverse_map = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=enable_diffusion_transverse_map",
        type=bool,
        cache=True,
        help="Use transverse diffusion map from field_dependencies_map_tmp",
    )

    # stupid naming problem...
    field_dependencies_map_tmp = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=field_dependencies_map"
        "&fmt=json.gz"
        "&method=WeightedNearestNeighbors",
        cache=True,
        help="Map for the electric field dependencies",
    )

    s2_mean_area_fraction_top = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=s2_mean_area_fraction_top",
        type=(int, float),
        cache=True,
        help="Mean S2 area fraction top",
    )

    s2_pattern_map = straxen.URLConfig(
        default="s2_aft_scaling://pattern_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=s2_pattern_map"
        "&fmt=pkl"
        "&pmt_mask=plugin.pmt_mask"
        "&s2_mean_area_fraction_top=plugin.s2_mean_area_fraction_top"
        "&n_tpc_pmts=plugin.n_tpc_pmts"
        "&n_top_pmts=plugin.n_top_pmts"
        "&turned_off_pmts=plugin.turned_off_pmts",
        cache=True,
        help="S2 pattern map",
    )

    singlet_fraction_gas = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=singlet_fraction_gas",
        type=(int, float),
        cache=True,
        help="Fraction of singlet states in GXe",
    )

    triplet_lifetime_gas = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=triplet_lifetime_gas",
        type=(int, float),
        cache=True,
        help="Liftetime of triplet states in GXe [ns]",
    )

    singlet_lifetime_gas = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=singlet_lifetime_gas",
        type=(int, float),
        cache=True,
        help="Liftetime of singlet states in GXe [ns]",
    )

    triplet_lifetime_liquid = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=triplet_lifetime_liquid",
        type=(int, float),
        cache=True,
        help="Liftetime of triplet states in LXe [ns]",
    )

    singlet_lifetime_liquid = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=singlet_lifetime_liquid",
        type=(int, float),
        cache=True,
        help="Liftetime of singlet states in LXe [ns]",
    )

    s2_secondary_sc_gain_mc = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=s2_secondary_sc_gain",
        type=(int, float),
        cache=True,
        help="Secondary scintillation gain [PE/e-]",
    )

    propagated_s2_photons_file_size_target = straxen.URLConfig(
        type=(int, float),
        default=300,
        track=False,
        help="Target for the propagated_s2_photons file size [MB]",
    )

    min_electron_gap_length_for_splitting = straxen.URLConfig(
        type=(int, float),
        default=2e6,
        track=False,
        help="Chunk can not be split if gap between photons is smaller than this value given in ns",
    )

    def setup(self):
        super().setup()

        # Set the random generator for scipy
        skewnorm.random_state = self.rng

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

        # Field dependencies
        if self.enable_diffusion_transverse_map:

            def rz_map(z, xy, **kwargs):
                r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
                return self.field_dependencies_map_tmp(np.array([r, z]).T, **kwargs)

            self.field_dependencies_map = rz_map

    def compute(self, interactions_in_roi, individual_electrons, start, end):
        # Just apply this to clusters with photons
        mask = interactions_in_roi["n_electron_extracted"] > 0

        if len(individual_electrons) == 0:
            yield self.chunk(start=start, end=end, data=np.zeros(0, dtype=self.dtype))
            return

        # Split into "sub-chunks"
        electron_time_gaps = individual_electrons["time"][1:] - individual_electrons["time"][:-1]
        electron_time_gaps = np.append(electron_time_gaps, 0)  # Add last gap

        split_index = find_electron_split_index(
            individual_electrons,
            electron_time_gaps,
            file_size_limit=self.propagated_s2_photons_file_size_target,
            min_gap_length=self.min_electron_gap_length_for_splitting,
            mean_n_photons_per_electron=self.s2_secondary_sc_gain_mc,
        )

        electron_chunks = np.array_split(individual_electrons, split_index)

        n_chunks = len(electron_chunks)
        if n_chunks > 1:
            self.log.info(
                f"Chunk size exceeding file size target. Downchunking to {n_chunks} chunks"
            )

        last_start = start
        for i, electron_group in enumerate(electron_chunks):
            result = self.compute_chunk(interactions_in_roi, mask, electron_group)

            # Move the chunk bound 90% of the minimal gap length to
            # the next photon to make space for afterpluses
            if i < n_chunks - 1:
                chunk_end = np.max(strax.endtime(result)) + np.int64(
                    self.min_electron_gap_length_for_splitting * 0.9
                )
            else:
                chunk_end = end
            chunk = self.chunk(start=last_start, end=chunk_end, data=result)
            last_start = chunk_end
            yield chunk

    def compute_chunk(self, interactions_in_roi, mask, electron_group):
        unique_clusters_in_group = np.unique(electron_group["cluster_id"])
        interactions_chunk = interactions_in_roi[mask][
            np.isin(interactions_in_roi["cluster_id"][mask], unique_clusters_in_group)
        ]

        # Sort both the interactions and the electrons by cluster_id
        # We will later sort by time again when yielding the data.
        sort_index_ic = np.argsort(interactions_chunk["cluster_id"])
        sort_index_eg = np.argsort(electron_group["cluster_id"])
        interactions_chunk = interactions_chunk[sort_index_ic]
        electron_group = electron_group[sort_index_eg]

        positions = np.array([interactions_chunk["x_obs"], interactions_chunk["y_obs"]]).T

        _photon_channels = self.photon_channels(
            interactions_chunk["n_electron_extracted"],
            interactions_chunk["z_obs"],
            positions,
            interactions_chunk["drift_time_mean"],
            interactions_chunk["sum_s2_photons"],
        )

        _photon_timings = self.photon_timings(
            positions,
            interactions_chunk["sum_s2_photons"],
            _photon_channels,
        ).astype(np.int64)

        # Repeat for n photons per electron # Should this be before adding delays?
        _photon_timings += np.repeat(electron_group["time"], electron_group["n_s2_photons"])

        _cluster_id = np.repeat(
            interactions_chunk["cluster_id"], interactions_chunk["sum_s2_photons"]
        )

        # Do i want to save both -> timings with and without pmt transit time spread?
        # Correct for PMT Transit Time Spread
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
            photon_type=2,
        )

        # Discard photons associated with negative channel numbers
        result = result[result["channel"] >= 0]

        result = strax.sort_by_time(result)

        return result

    def photon_channels(self, n_electron, z_obs, positions, drift_time_mean, n_photons):
        channels = np.arange(self.n_tpc_pmts).astype(np.int64)
        top_index = np.arange(self.n_top_pmts)
        bottom_index = np.arange(self.n_top_pmts, self.n_tpc_pmts)

        if self.diffusion_constant_transverse > 0:
            pattern = self.s2_pattern_map_diffuse(
                n_electron, z_obs, positions, drift_time_mean
            )  # [position, pmt]
        else:
            pattern = self.s2_pattern_map(positions)  # [position, pmt]

        if pattern.shape[1] - 1 not in bottom_index:
            pattern = np.pad(
                pattern, [[0, 0], [0, len(bottom_index)]], "constant", constant_values=1
            )

        sum_pat = np.sum(pattern, axis=1).reshape(-1, 1)
        pattern = np.divide(pattern, sum_pat, out=np.zeros_like(pattern), where=sum_pat != 0)

        assert pattern.shape[0] == len(positions)
        assert pattern.shape[1] == len(channels)

        _buffer_photon_channels = []
        # Randomly assign to channel given probability of each channel
        for i, n_ph in enumerate(n_photons):
            pat = pattern[i]

            # Redistribute pattern with user specified aft smearing
            if self.s2_aft_sigma != 0:
                _cur_aft = np.sum(pat[top_index]) / np.sum(pat)
                _new_aft = _cur_aft * skewnorm.rvs(
                    loc=1.0, scale=self.s2_aft_sigma, a=self.s2_aft_skewness
                )
                _new_aft = np.clip(_new_aft, 0, 1)
                pat[top_index] *= _new_aft / _cur_aft
                pat[bottom_index] *= (1 - _new_aft) / (1 - _cur_aft)

            # If pattern map return zeros or has NAN values assign negative channel
            # Photons with negative channel number will be rejected when
            # building photon propagation output
            if np.isnan(pat).sum() > 0:
                _photon_channels = np.array([-1] * n_ph)

            else:
                _photon_channels = self.rng.choice(channels, size=n_ph, p=pat, replace=True)

            _buffer_photon_channels.append(_photon_channels)

        _photon_channels = np.concatenate(_buffer_photon_channels)

        return _photon_channels.astype(np.int64)

    def s2_pattern_map_diffuse(self, n_electron, z, xy, drift_time_mean):
        """Returns an array of pattern of shape [n interaction, n PMTs] pattern
        of each interaction is an average of n_electron patterns evaluated at
        diffused position near xy.

        The diffused positions sample from 2d symmetric gaussian with
        spread scale with sqrt of drift time.
        Args:
            n_electron: a 1d int array
            z: a 1d float array
            xy: a 2d float array of shape [n interaction, 2]
            config: dict of the wfsim config
            resource: instance of the resource class
        """
        assert all(z < 0), "All S2 in liquid should have z < 0"

        if self.enable_diffusion_transverse_map:
            diffusion_constant_radial = self.field_dependencies_map(
                z, xy, map_name="diffusion_radial_map"
            )  # cm²/s
            diffusion_constant_azimuthal = self.field_dependencies_map(
                z, xy, map_name="diffusion_azimuthal_map"
            )  # cm²/s
            diffusion_constant_radial *= 1e-9  # cm²/ns
            diffusion_constant_azimuthal *= 1e-9  # cm²/ns
        else:
            diffusion_constant_radial = self.diffusion_constant_transverse
            diffusion_constant_azimuthal = self.diffusion_constant_transverse

        hdiff = np.zeros((np.sum(n_electron), 2))
        hdiff = simulate_horizontal_shift(
            n_electron,
            drift_time_mean,
            xy,
            diffusion_constant_radial,
            diffusion_constant_azimuthal,
            hdiff,
            self.rng,
        )

        # Should we also output this xy position in truth?
        xy_multi = np.repeat(xy, n_electron, axis=0) + hdiff  # One entry xy per electron
        # Remove points outside tpc, and the pattern will be the average inside tpc
        # TODO: Should be done naturally with the s2 pattern map, however,
        # there's some bug there, so we apply this hard cut
        mask = np.sum(xy_multi**2, axis=1) <= self.tpc_radius**2

        output_dim = self.s2_pattern_map.data["map"].shape[-1]

        pattern = np.zeros((len(n_electron), output_dim))
        n0 = 0
        # Average over electrons for each s2
        for ix, ne in enumerate(n_electron):
            s = slice(n0, n0 + ne)
            pattern[ix, :] = np.average(self.s2_pattern_map(xy_multi[s][mask[s]]), axis=0)
            n0 += ne

        return pattern

    def singlet_triplet_delays(self, size, singlet_ratio):
        """Given the amount of the excimer, return time between excimer decay.

        and their time of generation.
        size           - amount of excimer
        self.phase     - 'liquid' or 'gas'
        singlet_ratio  - fraction of excimers that become singlets
                         (NOT the ratio of singlets/triplets!)
        """
        if self.phase_s2 == "liquid":
            t1, t3 = (self.singlet_lifetime_liquid, self.triplet_lifetime_liquid)
        elif self.phase_s2 == "gas":
            t1, t3 = (self.singlet_lifetime_gas, self.triplet_lifetime_gas)
        else:
            t1, t3 = 0, 0

        delay = self.rng.choice([t1, t3], size, replace=True, p=[singlet_ratio, 1 - singlet_ratio])
        return (self.rng.exponential(1, size) * delay).astype(np.int64)

    def photon_timings(self, positions, n_photons, _photon_channels):
        raise NotImplementedError  # This is implemented in the child class


@export
class S2PhotonPropagation(S2PhotonPropagationBase):
    """This class is used to simulate the propagation of S2 photons using
    luminescence timing from garfield gas gap, singlet and tripled delays and
    optical propagation."""

    __version__ = "0.2.0"

    child_plugin = True

    s2_luminescence_map = straxen.URLConfig(
        default="simple_load://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=s2_luminescence_gg"
        "&fmt=npy",
        cache=True,
        help="Luminescence map for S2 Signals",
    )

    garfield_gas_gap_map = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=garfield_gas_gap_map"
        "&fmt=json",
        cache=True,
        help="Garfield gas gap map",
    )

    s2_optical_propagation_spline = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=s2_time_spline"
        "&fmt=json.gz"
        "&method=RegularGridInterpolator",
        cache=True,
        help="Spline for the optical propagation of S2 signals",
    )

    def setup(self):
        super().setup()
        self.log.debug(
            "Using Garfield GasGap luminescence timing and optical propagation "
            f"with plugin version {self.__version__}"
        )

    def photon_timings(self, positions, n_photons, _photon_channels):
        _photon_timings = self.luminescence_timings_garfield_gasgap(positions, n_photons)

        # Emission Delay
        _photon_timings += self.singlet_triplet_delays(
            len(_photon_timings), self.singlet_fraction_gas
        )

        # Optical Propagation Delay
        _photon_timings += self.optical_propagation(_photon_channels)

        return _photon_timings

    def luminescence_timings_garfield_gasgap(self, xy, n_photons):
        """Luminescence time distribution computation according to garfield
        scintillation maps which are ONLY drawn from below the anode, and at
        different gas gaps
        Args:
            xy: 1d array with positions
            n_photons: 1d array with ints for number of xy positions
        Returns:
            2d array with ints for photon timings of input param 'shape'
        """
        # assert 's2_luminescence_gg' in resource.__dict__, 's2_luminescence_gg model not found'
        assert len(n_photons) == len(
            xy
        ), "Input number of n_electron should have same length as positions"

        d_gasgap = self.s2_luminescence_map["gas_gap"][1] - self.s2_luminescence_map["gas_gap"][0]

        cont_gas_gaps = self.garfield_gas_gap_map(xy)
        draw_index = np.digitize(cont_gas_gaps, self.s2_luminescence_map["gas_gap"]) - 1
        diff_nearest_gg = cont_gas_gaps - self.s2_luminescence_map["gas_gap"][draw_index]

        return draw_excitation_times(
            self.s2_luminescence_map["timing_inv_cdf"],
            draw_index,
            n_photons,
            diff_nearest_gg,
            d_gasgap,
            self.rng,
        )

    def optical_propagation(self, channels):
        """Function getting times from s2 timing splines:
        Args:
            channels: The channels of all s2 photon
        """
        prop_time = np.zeros_like(channels)
        u_rand = self.rng.random(len(channels))[:, None]

        is_top = channels < self.n_top_pmts
        prop_time[is_top] = self.s2_optical_propagation_spline(u_rand[is_top], map_name="top")

        is_bottom = channels >= self.n_top_pmts
        prop_time[is_bottom] = self.s2_optical_propagation_spline(
            u_rand[is_bottom], map_name="bottom"
        )

        return prop_time.astype(np.int64)


@export
class S2PhotonPropagationSimple(S2PhotonPropagationBase):
    """This class is used to simulate the propagation of S2 photons using the
    simple liminescence model, singlet and tripled delays and optical
    propagation."""

    __version__ = "0.1.0"

    child_plugin = True

    pressure = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?&fmt=json&take=pressure",
        type=(int, float),
        cache=True,
        help="Pressure of liquid xenon [bar/e], while e is the elementary charge",
    )

    temperature = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?&fmt=json&take=temperature",
        type=(int, float),
        cache=True,
        help="Temperature of liquid xenon [K]",
    )

    gas_drift_velocity_slope = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=gas_drift_velocity_slope",
        type=(int, float),
        cache=True,
        help="gas_drift_velocity_slope",
    )

    enable_gas_gap_warping = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=enable_gas_gap_warping",
        type=bool,
        cache=True,
        help="enable_gas_gap_warping",
    )

    elr_gas_gap_length = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=elr_gas_gap_length",
        type=(int, float),
        cache=True,
        help="elr_gas_gap_length",
    )

    gas_gap_map = straxen.URLConfig(
        default="simple_load://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=gas_gap_map"
        "&fmt=pkl",
        cache=True,
        help="gas_gap_map",
    )

    anode_field_domination_distance = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=anode_field_domination_distance",
        type=(int, float),
        cache=True,
        help="anode_field_domination_distance",
    )

    anode_wire_radius = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=anode_wire_radius",
        type=(int, float),
        cache=True,
        help="anode_wire_radius",
    )

    gate_to_anode_distance = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=gate_to_anode_distance",
        type=(int, float),
        cache=True,
        help="Top of gate to bottom of anode (not considering perpendicular wires) [cm]",
    )

    anode_voltage = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?&fmt=json&take=anode_voltage",
        type=(int, float),
        cache=True,
        help="Voltage of anode [V]",
    )

    lxe_dielectric_constant = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=lxe_dielectric_constant",
        type=(int, float),
        cache=True,
        help="lxe_dielectric_constant",
    )

    s2_optical_propagation_spline = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=s2_time_spline"
        "&fmt=json.gz"
        "&method=RegularGridInterpolator",
        cache=True,
        help="Spline for the optical propagation of S2 signals",
    )

    def setup(self):
        super().setup()
        self.log.debug("Using simple luminescence timing and optical propagation")
        self.log.warn(
            "This is a legacy option, do you really want to use the simple luminescence model?"
        )

    def photon_timings(self, positions, n_photons, _photon_channels):
        _photon_timings = self.luminescence_timings_simple(positions, n_photons)

        # Emission Delay
        _photon_timings += self.singlet_triplet_delays(
            len(_photon_timings), self.singlet_fraction_gas
        )

        # Optical Propagation Delay
        _photon_timings += self.optical_propagation(_photon_channels)

        return _photon_timings

    def luminescence_timings_simple(self, xy, n_photons):
        """Luminescence time distribution computation according to simple s2
        model (many many many single electrons)
        Args:
            xy: 1d array with positions
            n_photons: 1d array with ints for number of xy positions
            config: dict wfsim config
            resource: instance of wfsim resource
        """
        assert len(n_photons) == len(
            xy
        ), "Input number of n_photons should have same length as positions"

        number_density_gas = self.pressure / (
            constants.Boltzmann / constants.elementary_charge * self.temperature
        )
        alpha = self.gas_drift_velocity_slope / number_density_gas
        uE = 1000 / 1  # V/cm
        pressure = self.pressure / conversion_to_bar

        if self.enable_gas_gap_warping:
            dG = self.gas_gap_map.lookup(*xy.T)
            dG = np.ma.getdata(dG)  # Convert from masked array to ndarray?
        else:
            dG = np.ones(len(xy)) * self.elr_gas_gap_length
        rA = self.anode_field_domination_distance
        rW = self.anode_wire_radius
        dL = self.gate_to_anode_distance - dG

        VG = self.anode_voltage / (1 + dL / dG / self.lxe_dielectric_constant)
        E0 = VG / ((dG - rA) / rA + np.log(rA / rW))  # V / cm

        dr = 0.0001  # cm
        r = np.arange(np.max(dG), rW, -dr)
        rr = np.clip(1 / r, 1 / rA, 1 / rW)

        return _luminescence_timings_simple(
            len(xy), dG, E0, r, dr, rr, alpha, uE, pressure, n_photons
        )

    def optical_propagation(self, channels):
        """Function getting times from s2 timing splines:
        Args:
            channels: The channels of all s2 photon
        """
        prop_time = np.zeros_like(channels)
        u_rand = self.rng.random(len(channels))[:, None]

        is_top = channels < self.n_top_pmts
        prop_time[is_top] = self.s2_optical_propagation_spline(u_rand[is_top], map_name="top")

        is_bottom = channels >= self.n_top_pmts
        prop_time[is_bottom] = self.s2_optical_propagation_spline(
            u_rand[is_bottom], map_name="bottom"
        )

        return prop_time.astype(np.int64)


@njit
def draw_excitation_times(inv_cdf_list, hist_indices, nph, diff_nearest_gg, d_gas_gap, rng):
    """Draws the excitation times from the GARFIELD electroluminescence map.

    Args:
        inv_cdf_list: List of inverse CDFs for the excitation time histograms
        hist_indices: The index of the histogram which refers to the gas gap
        nph: A 1-d array of the number of photons per electron
        diff_nearest_gg: The difference between the gas gap from the
            map (continuous value) and the nearest (discrete) value of the
            gas gap corresponding to the excitation time histograms
            d_gas_gap: Spacing between two consecutive gas gap values
    Returns:
        time of each photon
    """

    inv_cdf_len = len(inv_cdf_list[0])
    timings = np.zeros(np.sum(nph))
    upper_hist_ind = np.clip(hist_indices + 1, 0, len(inv_cdf_list) - 1)

    count = 0
    for i, (hist_ind, u_hist_ind, n, dngg) in enumerate(
        zip(hist_indices, upper_hist_ind, nph, diff_nearest_gg)
    ):
        # There are only 10 values of gas gap separated by 0.1mm, so we interpolate
        # between two histograms

        interp_cdf = (inv_cdf_list[u_hist_ind] - inv_cdf_list[hist_ind]) * (
            dngg / d_gas_gap
        ) + inv_cdf_list[hist_ind]

        # Subtract 2 because this way we don't want to sample from this last strange tail
        samples = rng.uniform(0, inv_cdf_len - 2, n)
        # samples = np.random.uniform(0, inv_cdf_len-2, n)
        t1 = interp_cdf[np.floor(samples).astype("int")]
        t2 = interp_cdf[np.ceil(samples).astype("int")]
        T = (t2 - t1) * (samples - np.floor(samples)) + t1
        if n != 0:
            T = T - np.mean(T)

        # subtract mean to get proper drift time and z correlation
        timings[count : count + n] = T
        count += n
    return timings


@njit
def _luminescence_timings_simple(
    n,
    dG,
    E0,
    r,
    dr,
    rr,
    alpha,
    uE,
    p,
    n_photons,
    rng,
):
    """Luminescence time distribution computation, calculates emission timings
    of photons from the excited electrons return 1d nested array with ints."""
    emission_time = np.zeros(np.sum(n_photons), np.int64)

    ci = 0
    for i in range(n):
        npho = n_photons[i]
        dt = dr / (alpha * E0[i] * rr)
        dy = E0[i] * rr / uE - 0.8 * p  # arXiv:physics/0702142
        avgt = np.sum(np.cumsum(dt) * dy) / np.sum(dy)

        j = np.argmax(r <= dG[i])
        t = np.cumsum(dt[j:]) - avgt
        y = np.cumsum(dy[j:])

        probabilities = rng.random(npho)
        emission_time[ci : ci + npho] = np.interp(probabilities, y / y[-1], t).astype(np.int64)
        ci += npho

    return emission_time


@njit()
def simulate_horizontal_shift(
    n_electron,
    drift_time_mean,
    xy,
    diffusion_constant_radial,
    diffusion_constant_azimuthal,
    result,
    rng,
):
    hdiff_stdev_radial = np.sqrt(2 * diffusion_constant_radial * drift_time_mean)
    hdiff_stdev_azimuthal = np.sqrt(2 * diffusion_constant_azimuthal * drift_time_mean)
    hdiff_radial = rng.normal(0, 1, np.sum(n_electron)) * np.repeat(hdiff_stdev_radial, n_electron)
    hdiff_azimuthal = rng.normal(0, 1, np.sum(n_electron)) * np.repeat(
        hdiff_stdev_azimuthal, n_electron
    )
    hdiff = np.column_stack((hdiff_radial, hdiff_azimuthal))
    theta = np.arctan2(xy[:, 1], xy[:, 0])

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    matrix = build_rotation_matrix(sin_theta, cos_theta)

    split_hdiff = np.split(hdiff, np.cumsum(n_electron))[:-1]

    start_idx = np.append([0], np.cumsum(n_electron)[:-1])
    stop_idx = np.cumsum(n_electron)

    for i in range(len(matrix)):
        result[start_idx[i] : stop_idx[i]] = np.ascontiguousarray(split_hdiff[i]) @ matrix[i]

    return result


@njit()
def build_rotation_matrix(sin_theta, cos_theta):
    matrix = np.zeros((len(sin_theta), 2, 2))
    matrix[:, 0, 0] = cos_theta
    matrix[:, 0, 1] = sin_theta
    matrix[:, 1, 0] = -sin_theta
    matrix[:, 1, 1] = cos_theta
    return matrix


@njit()
def find_electron_split_index(
    electrons, gaps, file_size_limit, min_gap_length, mean_n_photons_per_electron
):
    n_bytes_per_photon = 23  # 8 + 8 + 4 + 2 + 1

    data_size_mb = 0
    split_index = []

    for i, (e, g) in enumerate(zip(electrons, gaps)):
        # Assumes data is later saved as int16
        data_size_mb += n_bytes_per_photon * mean_n_photons_per_electron / 1e6

        if data_size_mb < file_size_limit:
            continue

        if g >= min_gap_length:
            data_size_mb = 0
            split_index.append(i)

    return np.array(split_index) + 1
