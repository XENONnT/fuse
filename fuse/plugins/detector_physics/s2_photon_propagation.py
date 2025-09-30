import strax
import straxen
import numpy as np

from numba import njit
from scipy.stats import skewnorm
from scipy import constants

from ...dtypes import propagated_photons_fields
from ...common import (
    stable_argsort,
    pmt_gains,
    build_photon_propagation_output,
    init_spe_scaling_factor_distributions,
    pmt_transit_time_spread,
    photon_gain_calculation,
)
from ...plugin import FuseBaseDownChunkingPlugin

export, __all__ = strax.exporter()

conversion_to_bar = 1 / constants.elementary_charge / 1e1


@export
class S2PhotonPropagationBase(FuseBaseDownChunkingPlugin):
    """Base plugin to simulate the propagation of S2 photons in the
    detector."""

    __version__ = "0.4.2"

    depends_on = (
        "merged_s2_photons",
        "merged_extracted_electrons",
    )

    provides = "propagated_s2_photons"
    data_kind = "s2_photons"

    save_when = strax.SaveWhen.TARGET

    dtype = propagated_photons_fields + strax.time_fields

    # Shared config
    p_double_pe_emision = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?"
        "&fmt=json&take=p_double_pe_emision",
        type=(int, float),
        cache=True,
        help="Probability of double photo-electron emission",
    )
    pmt_transit_time_spread = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?"
        "&fmt=json&take=pmt_transit_time_spread",
        type=(int, float),
        cache=True,
        help="Spread of the PMT transit times [ns]",
    )
    pmt_transit_time_mean = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?"
        "&fmt=json&take=pmt_transit_time_mean",
        type=(int, float),
        cache=True,
        help="Mean of the PMT transit times [ns]",
    )
    pmt_circuit_load_resistor = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?"
        "&fmt=json&take=pmt_circuit_load_resistor",
        type=(int, float),
        cache=True,
        help="PMT circuit load resistor [kg m^2/(s^3 A)]",
    )
    digitizer_bits = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?" "&fmt=json&take=digitizer_bits",
        type=(int, float),
        cache=True,
        help="Number of bits of the digitizer boards",
    )
    digitizer_voltage_range = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?"
        "&fmt=json&take=digitizer_voltage_range",
        type=(int, float),
        cache=True,
        help="Voltage range of the digitizer boards [V]",
    )
    n_top_pmts = straxen.URLConfig(type=int, help="Number of PMTs on top array")
    n_tpc_pmts = straxen.URLConfig(type=int, help="Number of PMTs in the TPC")
    gain_model_mc = straxen.URLConfig(
        default=(
            "list-to-array://xedocs://pmt_area_to_pes"
            "?as_list=True&sort=pmt&detector=tpc"
            "&run_id=plugin.run_id&version=ONLINE&attr=value"
        ),
        infer_type=False,
        help="PMT gain model",
    )
    photon_area_distribution = straxen.URLConfig(
        default="simple_load://resource://simulation_config://SIMULATION_CONFIG_FILE.json?"
        "&key=photon_area_distribution&fmt=csv",
        cache=True,
        help="Photon area distribution",
    )

    # S2 specific configs
    phase_s2 = straxen.URLConfig(default="gas", help="Phase of the S2 producing region")
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
    s2_aft_skewness = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?&fmt=json&take=s2_aft_skewness",
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
    field_dependencies_map_tmp = straxen.URLConfig(
        default="itp_map://resource://simulation_config://SIMULATION_CONFIG_FILE.json?"
        "&key=field_dependencies_map&fmt=json.gz&method=WeightedNearestNeighbors",
        cache=True,
        help="Map for the electric field dependencies",
    )
    s2_mean_area_fraction_top = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?"
        "&fmt=json&take=s2_mean_area_fraction_top",
        type=(int, float),
        cache=True,
        help="Mean S2 area fraction top",
    )
    s2_pattern_map = straxen.URLConfig(
        default=(
            "s2_aft_scaling://pattern_map://resource://simulation_config://"
            "SIMULATION_CONFIG_FILE.json?&key=s2_pattern_map&fmt=pkl"
            "&pmt_mask=plugin.pmt_mask"
            "&s2_mean_area_fraction_top=plugin.s2_mean_area_fraction_top"
            "&n_tpc_pmts=plugin.n_tpc_pmts"
            "&n_top_pmts=plugin.n_top_pmts"
            "&turned_off_pmts=plugin.turned_off_pmts"
            "&method=plugin.s2_pattern_map_interpolation_method"
        ),
        cache=True,
        help="S2 pattern map",
    )
    s2_pattern_map_interpolation_method = straxen.URLConfig(
        default="WeightedNearestNeighbors",
        help="Interpolation method for the S2 pattern map",
        type=str,
        cache=True,
    )
    singlet_fraction_gas = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?"
        "&fmt=json&take=singlet_fraction_gas",
        type=(int, float),
        cache=True,
        help="Fraction of singlet states in GXe",
    )
    triplet_lifetime_gas = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?"
        "&fmt=json&take=triplet_lifetime_gas",
        type=(int, float),
        cache=True,
        help="Liftetime of triplet states in GXe [ns]",
    )
    singlet_lifetime_gas = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?"
        "&fmt=json&take=singlet_lifetime_gas",
        type=(int, float),
        cache=True,
        help="Liftetime of singlet states in GXe [ns]",
    )
    triplet_lifetime_liquid = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?"
        "&fmt=json&take=triplet_lifetime_liquid",
        type=(int, float),
        cache=True,
        help="Liftetime of triplet states in LXe [ns]",
    )
    singlet_lifetime_liquid = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?"
        "&fmt=json&take=singlet_lifetime_liquid",
        type=(int, float),
        cache=True,
        help="Liftetime of singlet states in LXe [ns]",
    )
    s2_secondary_sc_gain_mc = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?"
        "&fmt=json&take=s2_secondary_sc_gain",
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
        help="Chunk can not be split if gap between photons is smaller than this value [ns]",
    )

    def setup(self):
        super().setup()

        skewnorm.random_state = self.rng

        self.gains = pmt_gains(
            self.gain_model_mc,
            digitizer_voltage_range=self.digitizer_voltage_range,
            digitizer_bits=self.digitizer_bits,
            pmt_circuit_load_resistor=self.pmt_circuit_load_resistor,
        )

        self.pmt_mask = np.array(self.gains) > 0
        self.turned_off_pmts = np.nonzero(np.array(self.gains) == 0)[0]

        self.spe_scaling_factor_distributions = init_spe_scaling_factor_distributions(
            self.photon_area_distribution
        )

        # Precompute top/bottom indices
        self._top_idx = np.arange(self.n_top_pmts)
        self._bot_idx = np.arange(self.n_top_pmts, self.n_tpc_pmts)

        self._mem_cap_bytes = int(256 * 1024**2)  # ~256 MB for Ne×C scratch
        self._max_electrons_block = None  # computed lazily per chunk

    def compute(self, individual_electrons, start, end):

        if len(individual_electrons) == 0:
            yield self.chunk(start=start, end=end, data=np.zeros(0, dtype=self.dtype))
            return

        # Downchunking (logic unchanged)
        electron_time_gaps = individual_electrons["time"][1:] - individual_electrons["time"][:-1]
        electron_time_gaps = np.append(electron_time_gaps, 0)

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
            result = self.compute_chunk(electron_group)

            if i < n_chunks - 1:
                chunk_end = np.max(strax.endtime(result)) + np.int64(
                    self.min_electron_gap_length_for_splitting * 0.9
                )
            else:
                chunk_end = end
            chunk = self.chunk(start=last_start, end=chunk_end, data=result)
            last_start = chunk_end
            yield chunk

    def compute_chunk(self, electron_group):

        sort_index_eg = stable_argsort(electron_group["cluster_id"])
        eg = electron_group[sort_index_eg]

        pos = np.column_stack((eg["x_interface"], eg["y_interface"]))
        nph = eg["n_s2_photons"]

        # Channel sampling
        out_ch = self.photon_channels(pos, nph)

        # Time sampling
        t_rel = self.photon_timings(pos, nph, out_ch)

        # Add electron absolute times per-photon
        t_rel += np.repeat(eg["time"], nph)

        # transit spread
        t_rel = pmt_transit_time_spread(
            _photon_timings=t_rel,
            pmt_transit_time_mean=self.pmt_transit_time_mean,
            pmt_transit_time_spread=self.pmt_transit_time_spread,
            rng=self.rng,
        )

        # gains/DPE
        g, is_dpe = photon_gain_calculation(
            _photon_channels=out_ch,
            p_double_pe_emision=self.p_double_pe_emision,
            gains=self.gains,
            spe_scaling_factor_distributions=self.spe_scaling_factor_distributions,
            rng=self.rng,
        )

        # cluster ids per photon
        cl_ids = np.repeat(eg["cluster_id"], nph)

        res = build_photon_propagation_output(
            dtype=self.dtype,
            _photon_timings=t_rel,
            _photon_channels=out_ch,
            _photon_gains=g,
            _photon_is_dpe=is_dpe,
            _cluster_id=cl_ids,
            photon_type=2,
        )
        res = res[res["channel"] >= 0]
        res = strax.sort_by_time(res)
        return res

    def photon_channels(self, positions, n_photons):
        """Blocked, memory-capped channel sampling:

        1) pattern map -> pad to full TPC PMTs if needed 2) optional AFT
        smearing (skewnorm) — unchanged math 3) row-normalize 4) in-
        place CDF (cumsum) 5) per-row searchsorted for uniforms Returns
        int16 array of length sum(n_photons).
        """
        total = int(np.sum(n_photons))
        out = np.empty(total, dtype=np.int16)

        # decide electron block size from mem cap: one (Ne x C) buffer
        C_full = self.n_tpc_pmts
        bytes_per = 4
        # default to ~128MB if not set
        mem_cap = getattr(self, "_mem_cap_bytes", 128 * (1024**2))
        max_ne = max(1, mem_cap // (C_full * bytes_per))

        # prefix-map electron -> photon slice [start, stop)
        starts = np.empty(len(n_photons), np.int64)
        stops = np.empty(len(n_photons), np.int64)
        s = 0
        for i, k in enumerate(n_photons):
            starts[i] = s
            s += int(k)
            stops[i] = s

        # process electrons in blocks
        for i0 in range(0, len(positions), max_ne):
            i1 = min(i0 + max_ne, len(positions))
            pos_blk = positions[i0:i1]

            # 1) pattern map
            pattern = self.s2_pattern_map(pos_blk)
            pattern = np.ascontiguousarray(pattern)

            # pad to full TPC PMTs if pattern missing bottoms
            C = pattern.shape[1]
            if C < C_full:
                pat2 = np.zeros((pattern.shape[0], C_full), dtype=np.float32)
                pat2[:, :C] = pattern
                pattern = pat2

            # 2) AFT smearing (unchanged math)
            if self.s2_aft_sigma != 0:
                sum_top = pattern[:, self._top_idx].sum(axis=1)
                sum_all = pattern.sum(axis=1)
                cur_aft = np.divide(
                    sum_top, sum_all, out=np.zeros_like(sum_top), where=sum_all != 0
                )

                new_aft = cur_aft * skewnorm.rvs(
                    loc=1.0, scale=self.s2_aft_sigma, a=self.s2_aft_skewness, size=cur_aft.shape[0]
                )
                new_aft = np.clip(new_aft, 0.0, 1.0)

                with np.errstate(divide="ignore", invalid="ignore"):
                    scale_top = np.divide(
                        new_aft, cur_aft, out=np.ones_like(new_aft), where=cur_aft > 0
                    )
                    scale_bot = np.divide(
                        1 - new_aft, 1 - cur_aft, out=np.ones_like(new_aft), where=cur_aft < 1
                    )

                pattern[:, self._top_idx] *= scale_top[:, None]
                pattern[:, self._bot_idx] *= scale_bot[:, None]

            # 3) normalize rows
            row_sum = pattern.sum(axis=1, keepdims=True, dtype=np.float32)
            np.divide(pattern, row_sum, out=pattern, where=row_sum != 0)

            # 4) CDF in-place
            np.cumsum(pattern, axis=1, out=pattern)
            pattern[:, -1] = 1.0  # exact

            # 5) per-electron sampling into global 'out'
            for ii in range(i0, i1):
                s = starts[ii]
                e = stops[ii]
                if s == e:
                    continue
                u = self.rng.random(e - s)  # [0,1)
                row = pattern[ii - i0]
                out[s:e] = np.searchsorted(row, u, side="left")

        return out

    def singlet_triplet_delays(self, size, singlet_ratio):
        if self.phase_s2 == "liquid":
            t1, t3 = (self.singlet_lifetime_liquid, self.triplet_lifetime_liquid)
        elif self.phase_s2 == "gas":
            t1, t3 = (self.singlet_lifetime_gas, self.triplet_lifetime_gas)
        else:
            t1, t3 = 0, 0

        delay = self.rng.choice([t1, t3], size, replace=True, p=[singlet_ratio, 1 - singlet_ratio])
        return self.rng.exponential(1, size) * delay

    def photon_timings(self, positions, n_photons, _photon_channels):
        raise NotImplementedError


@export
class S2PhotonPropagation(S2PhotonPropagationBase):
    """S2 photon propagation using Garfield gas-gap luminescence + optical
    propagation."""

    __version__ = "0.2.0"

    child_plugin = True

    s2_luminescence_map = straxen.URLConfig(
        default="simple_load://resource://simulation_config://SIMULATION_CONFIG_FILE.json?"
        "&key=s2_luminescence_gg&fmt=npy",
        cache=True,
        help="Luminescence map for S2 Signals",
    )

    garfield_gas_gap_map = straxen.URLConfig(
        default="itp_map://resource://simulation_config://SIMULATION_CONFIG_FILE.json?"
        "&key=garfield_gas_gap_map&fmt=json",
        cache=True,
        help="Garfield gas gap map",
    )

    s2_optical_propagation_spline = straxen.URLConfig(
        default="itp_map://resource://simulation_config://SIMULATION_CONFIG_FILE.json?"
        "&key=s2_time_spline&fmt=json.gz&method=RegularGridInterpolator",
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
        _photon_timings += self.singlet_triplet_delays(
            len(_photon_timings), self.singlet_fraction_gas
        )
        _photon_timings += self.optical_propagation(_photon_channels)
        return _photon_timings

    def luminescence_timings_garfield_gasgap(self, xy, n_photons):
        assert len(n_photons) == len(xy), "n_photons length must match positions"
        d_gasgap = self.s2_luminescence_map["gas_gap"][1] - self.s2_luminescence_map["gas_gap"][0]

        cont_gas_gaps = self.garfield_gas_gap_map(xy)
        draw_index = np.digitize(cont_gas_gaps, self.s2_luminescence_map["gas_gap"]) - 1
        diff_nearest_gg = cont_gas_gaps - self.s2_luminescence_map["gas_gap"][draw_index]

        total = int(np.sum(n_photons))
        inv_len = len(self.s2_luminescence_map["timing_inv_cdf"][0])
        samples_flat = self.rng.uniform(0.0, inv_len - 2, size=total)

        return draw_excitation_times(
            self.s2_luminescence_map["timing_inv_cdf"],
            draw_index,
            n_photons,
            diff_nearest_gg,
            float(d_gasgap),
            samples_flat,
        )

    def optical_propagation(self, channels):
        # Allocate as int64 (ns) so times are never truncated by int16 channels dtype
        prop_time = np.zeros(channels.shape[0], dtype=np.int64)

        # Spline expects shape (N, 1); behavior unchanged
        u_rand = self.rng.random(len(channels))[:, None]

        is_top = channels < self.n_top_pmts
        if is_top.any():
            prop_time[is_top] = self.s2_optical_propagation_spline(u_rand[is_top], map_name="top")

        is_bottom = ~is_top
        if is_bottom.any():
            prop_time[is_bottom] = self.s2_optical_propagation_spline(
                u_rand[is_bottom], map_name="bottom"
            )

        return prop_time


@export
class S2PhotonPropagationSimple(S2PhotonPropagationBase):
    """S2 photon propagation using simple luminescence model + optical
    propagation."""

    __version__ = "0.1.1"  # unchanged physics; perf tweaks only

    child_plugin = True

    pressure = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?" "&fmt=json&take=pressure",
        type=(int, float),
        cache=True,
        help="Pressure of liquid xenon [bar/e]",
    )
    temperature = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?" "&fmt=json&take=temperature",
        type=(int, float),
        cache=True,
        help="Temperature of liquid xenon [K]",
    )
    gas_drift_velocity_slope = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?"
        "&fmt=json&take=gas_drift_velocity_slope",
        type=(int, float),
        cache=True,
        help="gas_drift_velocity_slope",
    )
    enable_gas_gap_warping = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?"
        "&fmt=json&take=enable_gas_gap_warping",
        type=bool,
        cache=True,
        help="enable_gas_gap_warping",
    )
    elr_gas_gap_length = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?"
        "&fmt=json&take=elr_gas_gap_length",
        type=(int, float),
        cache=True,
        help="elr_gas_gap_length",
    )
    gas_gap_map = straxen.URLConfig(
        default="simple_load://resource://simulation_config://SIMULATION_CONFIG_FILE.json?"
        "&key=gas_gap_map&fmt=pkl",
        cache=True,
        help="gas_gap_map",
    )
    anode_field_domination_distance = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?"
        "&fmt=json&take=anode_field_domination_distance",
        type=(int, float),
        cache=True,
        help="anode_field_domination_distance",
    )
    anode_wire_radius = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?" "&fmt=json&take=anode_wire_radius",
        type=(int, float),
        cache=True,
        help="anode_wire_radius",
    )
    gate_to_anode_distance = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?"
        "&fmt=json&take=gate_to_anode_distance",
        type=(int, float),
        cache=True,
        help="Top of gate to bottom of anode [cm]",
    )
    anode_voltage = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?" "&fmt=json&take=anode_voltage",
        type=(int, float),
        cache=True,
        help="Voltage of anode [V]",
    )
    lxe_dielectric_constant = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?"
        "&fmt=json&take=lxe_dielectric_constant",
        type=(int, float),
        cache=True,
        help="lxe_dielectric_constant",
    )
    s2_optical_propagation_spline = straxen.URLConfig(
        default="itp_map://resource://simulation_config://SIMULATION_CONFIG_FILE.json?"
        "&key=s2_time_spline&fmt=json.gz&method=RegularGridInterpolator",
        cache=True,
        help="Spline for the optical propagation of S2 signals",
    )

    def setup(self):
        super().setup()
        self.log.debug("Using simple luminescence timing and optical propagation")
        self.log.warn("This is a legacy option; consider the Garfield model for realism.")

    def photon_timings(self, positions, n_photons, _photon_channels):
        _photon_timings = self.luminescence_timings_simple(positions, n_photons)
        _photon_timings += self.singlet_triplet_delays(
            len(_photon_timings), self.singlet_fraction_gas
        )
        _photon_timings += self.optical_propagation(_photon_channels)
        return _photon_timings

    def luminescence_timings_simple(self, xy, n_photons):
        assert len(n_photons) == len(xy), "n_photons length must match positions"

        number_density_gas = self.pressure / (
            constants.Boltzmann / constants.elementary_charge * self.temperature
        )
        alpha = self.gas_drift_velocity_slope / number_density_gas
        uE = 1000 / 1  # V/cm
        pressure = self.pressure / conversion_to_bar

        if self.enable_gas_gap_warping:
            dG = self.gas_gap_map.lookup(*xy.T)
            dG = np.ma.getdata(dG)
        else:
            dG = np.ones(len(xy)) * self.elr_gas_gap_length
        rA = self.anode_field_domination_distance
        rW = self.anode_wire_radius
        dL = self.gate_to_anode_distance - dG

        VG = self.anode_voltage / (1 + dL / dG / self.lxe_dielectric_constant)
        E0 = VG / ((dG - rA) / rA + np.log(rA / rW))  # V / cm

        dr = 0.0001  # cm
        # PERF: build descending r once using max(dG) (same as before)
        r = np.arange(np.max(dG), rW, -dr)
        rr = np.clip(1 / r, 1 / rA, 1 / rW)

        total = int(np.sum(n_photons))
        uniforms_flat = self.rng.random(total)

        return _luminescence_timings_simple(
            len(xy),
            dG,
            E0,
            r,
            float(dr),
            rr,
            float(alpha),
            float(uE),
            float(pressure),
            n_photons,
            uniforms_flat,
        )

    def optical_propagation(self, channels):
        # Allocate as int64 (ns) so times are never truncated by int16 channels dtype
        prop_time = np.zeros(channels.shape[0], dtype=np.int64)

        # Spline expects shape (N, 1); behavior unchanged
        u_rand = self.rng.random(len(channels))[:, None]

        is_top = channels < self.n_top_pmts
        if is_top.any():
            prop_time[is_top] = self.s2_optical_propagation_spline(u_rand[is_top], map_name="top")

        is_bottom = ~is_top
        if is_bottom.any():
            prop_time[is_bottom] = self.s2_optical_propagation_spline(
                u_rand[is_bottom], map_name="bottom"
            )

        return prop_time


@njit(cache=True)
def find_electron_split_index(
    electrons, gaps, file_size_limit, min_gap_length, mean_n_photons_per_electron
):
    # same byte accounting; keep structure/thresholds identical
    n_bytes_per_photon = 23  # 8 + 8 + 4 + 2 + 1

    data_size_mb = 0.0
    # Numba-friendly dynamic buffer: worst case = len(gaps)
    split_index = np.empty(gaps.shape[0], np.int64)
    n_splits = 0

    for i in range(gaps.shape[0]):
        data_size_mb += n_bytes_per_photon * mean_n_photons_per_electron / 1e6
        if data_size_mb < file_size_limit:
            # continue path
            pass
        else:
            if gaps[i] >= min_gap_length:
                data_size_mb = 0.0
                split_index[n_splits] = i
                n_splits += 1

    # trim to actual count and +1 as in original
    out = np.empty(n_splits, np.int64)
    for k in range(n_splits):
        out[k] = split_index[k] + 1
    return out


@njit(cache=True, fastmath=True)
def _segment_ranges(n_per_row):
    m = n_per_row.shape[0]
    starts = np.empty(m, np.int64)
    stops = np.empty(m, np.int64)
    total = 0
    for i in range(m):
        k = int(n_per_row[i])
        starts[i] = total
        total += k
        stops[i] = total
    return starts, stops, total


@njit(cache=True, fastmath=True)
def draw_excitation_times(
    inv_cdf_list, hist_indices, nph, diff_nearest_gg, d_gas_gap, samples_flat
):
    inv_cdf_len = len(inv_cdf_list[0])
    timings = np.zeros(np.sum(nph))
    upper_hist_ind = np.clip(hist_indices + 1, 0, len(inv_cdf_list) - 1)

    starts, stops, _ = _segment_ranges(nph)

    for i in range(nph.shape[0]):
        hist_ind = hist_indices[i]
        u_hist_ind = upper_hist_ind[i]
        n = int(nph[i])
        if n == 0:
            continue

        dngg = diff_nearest_gg[i]
        # linear interpolation between nearest inv-CDFs (same as before)
        interp_cdf = (inv_cdf_list[u_hist_ind] - inv_cdf_list[hist_ind]) * (
            dngg / d_gas_gap
        ) + inv_cdf_list[hist_ind]

        s = starts[i]
        e = stops[i]
        samples = samples_flat[s:e]  # in [0, inv_cdf_len-2)

        fi = np.floor(samples).astype(np.int64)
        # safe ceil capped at last valid index
        ci = fi + (fi < (inv_cdf_len - 2))
        t1 = interp_cdf[fi]
        t2 = interp_cdf[ci]
        T = (t2 - t1) * (samples - fi) + t1
        # preserve original zero-mean re-centering per electron
        T = T - T.mean()
        timings[s:e] = T
    return timings


@njit(cache=True, fastmath=True)
def _luminescence_timings_simple(
    n, dG, E0, r, dr, rr, alpha, uE, p, n_photons, uniforms_flat  # in [0,1)
):
    emission_time = np.zeros(np.sum(n_photons), np.int64)
    starts, stops, _ = _segment_ranges(n_photons)

    for i in range(n):
        npho = int(n_photons[i])
        if npho == 0:
            continue

        # identical algebra, just hoisted into numba
        dt_i = dr / (alpha * E0[i] * rr)
        dy_i = E0[i] * rr / uE - 0.8 * p  # arXiv:physics/0702142
        avgt = np.sum(np.cumsum(dt_i) * dy_i) / np.sum(dy_i)

        j = 0
        # first r <= dG[i]
        while j < r.shape[0] and not (r[j] <= dG[i]):
            j += 1

        t = np.cumsum(dt_i[j:]) - avgt
        y = np.cumsum(dy_i[j:])

        s = starts[i]
        e = stops[i]
        probs = uniforms_flat[s:e]
        # linear interp on monotonic CDF y/y[-1] → t
        emission_time[s:e] = np.interp(probs, y / y[-1], t)

    return emission_time
