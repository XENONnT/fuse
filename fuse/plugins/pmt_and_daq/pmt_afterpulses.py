import strax
import numpy as np
import numba
import straxen

from ...dtypes import propagated_photons_fields
from ...common import stable_argsort, pmt_gains
from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()


@export
class PMTAfterPulses(FuseBasePlugin):
    """Plugin to simulate PMT afterpulses using a precomputed afterpulse
    cumulative distribution function.

    In the simulation afterpulses will be saved as a list of "pseudo"
    photons. These "photons" can then be combined with real photons from
    S1 and S2 signals to create a waveform.
    """

    __version__ = "0.3.1"

    depends_on = ("propagated_s2_photons", "propagated_s1_photons")
    provides = "pmt_afterpulses"
    data_kind = "ap_photons"

    save_when = strax.SaveWhen.TARGET

    dtype = propagated_photons_fields + strax.time_fields

    # Config options

    enable_pmt_afterpulses = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=enable_pmt_afterpulses",
        type=bool,
        track=True,
        help="Decide if you want to to enable PMT afterpulsing",
    )

    pmt_ap_t_modifier = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=pmt_ap_t_modifier",
        type=(int, float),
        cache=True,
        help="PMT afterpulse time modifier",
    )

    pmt_ap_modifier = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=pmt_ap_modifier",
        type=(int, float),
        cache=True,
        help="PMT afterpulse modifier",
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
        default=(
            "list-to-array://xedocs://pmt_area_to_pes"
            "?as_list=True&sort=pmt&detector=tpc"
            "&run_id=plugin.run_id&version=ONLINE&attr=value"
        ),
        infer_type=False,
        help="PMT gain model",
    )

    photon_ap_cdfs = straxen.URLConfig(
        default="simple_load://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=photon_ap_cdfs"
        "&fmt=json.gz",
        cache=True,
        help="Afterpuse cumulative distribution functions",
    )

    def setup(self):
        super().setup()

        self.gains = pmt_gains(
            self.gain_model_mc,
            digitizer_voltage_range=self.digitizer_voltage_range,
            digitizer_bits=self.digitizer_bits,
            pmt_circuit_load_resistor=self.pmt_circuit_load_resistor,
        )

        self.uniform_to_pmt_ap = self.photon_ap_cdfs

        for k in self.uniform_to_pmt_ap.keys():
            for q in self.uniform_to_pmt_ap[k].keys():
                if isinstance(self.uniform_to_pmt_ap[k][q], list):
                    self.uniform_to_pmt_ap[k][q] = np.array(self.uniform_to_pmt_ap[k][q])

        # Pre-flatten the nested CDF dict into parallel arrays for the per-element
        # loop. Cast every CDF to float64 once here so the numba kernel sees a
        # single dispatch specialisation and per-iteration arithmetic uses a
        # fixed dtype throughout (avoids per-chunk promotions).
        self._ap_elements = list(self.uniform_to_pmt_ap.keys())
        self._ap_is_uniform = np.array(
            [("Uniform" in e) for e in self._ap_elements], dtype=np.bool_
        )
        self._ap_delaytime_cdf = [
            np.asarray(self.uniform_to_pmt_ap[e]["delaytime_cdf"], dtype=np.float64)
            for e in self._ap_elements
        ]
        self._ap_amplitude_cdf = [
            np.asarray(self.uniform_to_pmt_ap[e]["amplitude_cdf"], dtype=np.float64)
            for e in self._ap_elements
        ]
        self._ap_amplitude_cdf_ndim = np.array(
            [cdf.ndim for cdf in self._ap_amplitude_cdf], dtype=np.int8
        )
        self._ap_delaytime_bin_size = np.array(
            [self.uniform_to_pmt_ap[e]["delaytime_bin_size"] for e in self._ap_elements],
            dtype=np.float64,
        )
        self._ap_amplitude_bin_size = np.array(
            [self.uniform_to_pmt_ap[e]["amplitude_bin_size"] for e in self._ap_elements],
            dtype=np.float64,
        )
        self._gains_f64 = np.asarray(self.gains, dtype=np.float64)

    def compute(self, s1_photons, s2_photons):
        if not self.enable_pmt_afterpulses or (len(s1_photons) == 0 and len(s2_photons) == 0):
            return np.zeros(0, dtype=self.dtype)

        merged_photons = np.concatenate([s1_photons, s2_photons])
        s1_photons = None
        s2_photons = None

        # Sort all photons by time
        sortind = stable_argsort(merged_photons["time"])
        merged_photons = merged_photons[sortind]

        _photon_timings = merged_photons["time"]
        _photon_channels = merged_photons["channel"]
        _photon_is_dpe = merged_photons["dpe"]

        ap_photon_timings, ap_photon_channels, ap_photon_gains = self.photon_afterpulse(
            _photon_timings, _photon_channels, _photon_is_dpe
        )
        ap_photon_is_dpe = np.zeros_like(ap_photon_timings).astype(np.bool_)

        result = np.zeros(len(ap_photon_channels), dtype=self.dtype)
        result["channel"] = ap_photon_channels
        result["time"] = ap_photon_timings
        result["endtime"] = ap_photon_timings
        result["dpe"] = ap_photon_is_dpe
        result["photon_gain"] = ap_photon_gains

        result["cluster_id"] = -1 * np.ones(len(ap_photon_channels))

        result = strax.sort_by_time(result)

        return result

    def photon_afterpulse(
        self, merged_photon_timings, merged_photon_channels, merged_photon_id_dpe
    ):
        """For pmt afterpulses, gain and dpe generation is a bit different from
        standard photons.

        Two-pass numba scheme for the per-element afterpulse loop. Random
        draws stay in Python and follow a fixed per-element order — same
        argument shapes and same conditional skip on empty selection — so
        the PCG64 sequence is preserved across calls.

        Per element:
          1. ``rng.random(N)`` for rU0 (Python).
          2. ``_ap_select_kernel`` (numba): one tight loop that scales
             rU0 in place by the per-element modifier (and halves the
             dpe entries), compares against ``cdf[channel, -1]`` for
             each row, and emits the indices that pass the selection
             plus the running max of the per-row probability ceiling.
          3. ``rng.random(N_sel)`` for rU1 (Python; skipped when
             ``N_sel == 0`` — the skip MUST stay before this draw or
             the random sequence diverges).
          4. Uniform branch: ``rng.uniform(low, high)`` (Python; only
             on Uniform elements). Non-Uniform branch:
             ``_ap_kernel_nonuniform`` does the argmin-based inverse
             CDF lookup.

        Every float operation matches the equivalent numpy expression's
        IEEE-754 sequence; rng calls happen in a deterministic order
        with the same argument shapes per element.
        """
        _photon_timings = []
        _photon_channels = []
        _photon_gains = []

        N = len(merged_photon_timings)
        # Allocate the selection-index scratch buffer ONCE per call. Reused
        # across all elements; only the first n_sel entries are valid per element.
        sel_buf = np.empty(N, dtype=np.int64)
        # Coerce channels to int64 once outside the per-element loop.
        chans_int = merged_photon_channels.astype(np.int64, copy=False)
        modifier = float(self.pmt_ap_modifier)

        for i, element in enumerate(self._ap_elements):
            delaytime_cdf = self._ap_delaytime_cdf[i]
            amplitude_cdf = self._ap_amplitude_cdf[i]
            delaytime_bin_size = self._ap_delaytime_bin_size[i]
            amplitude_bin_size = self._ap_amplitude_bin_size[i]

            # First rng draw for this element. Must happen unconditionally
            # before the selection branch — moving it inside the branch
            # would diverge the PCG64 state on elements with zero
            # selections.
            rU0 = 1 - self.rng.random(N)

            # Pass 1 (numba): scales rU0 in-place, emits selection indices,
            # also returns max(prob_ap) for the warning check.
            n_sel, prob_max = _ap_select_kernel(
                rU0,
                chans_int,
                merged_photon_id_dpe,
                delaytime_cdf,
                modifier,
                sel_buf,
            )

            if prob_max * modifier > 0.5:
                prob = prob_max * modifier
                self.log.warning(f"PMT after pulse probability is {prob} larger than 0.5?")

            if n_sel == 0:
                continue

            sel_photon_id = sel_buf[:n_sel]
            sel_photon_channel = chans_int[sel_photon_id]

            # Second rng draw. The `continue` on n_sel == 0 above MUST stay
            # before this draw — skipping it would diverge the PCG64 state
            # on elements with empty selections.
            rU1 = 1 - self.rng.random(n_sel)

            if self._ap_is_uniform[i]:
                # Third rng draw, conditional on the Uniform branch.
                # `rng.uniform(low, high)` consumes len(sel) uniforms internally.
                ap_delay = (
                    self.rng.uniform(
                        delaytime_cdf[sel_photon_channel, 0],
                        delaytime_cdf[sel_photon_channel, 1],
                    )
                    * delaytime_bin_size
                )
                ap_amplitude = np.ones_like(ap_delay)
            else:
                # Non-Uniform branch: argmin-based inverse CDF in the kernel
                # below.
                rU0_sel = rU0[sel_photon_id]
                amp_ndim = int(self._ap_amplitude_cdf_ndim[i])
                if amp_ndim == 2:
                    amp_cdf_2d = amplitude_cdf
                    amp_cdf_1d = _AP_1D_PLACEHOLDER
                else:
                    amp_cdf_2d = _AP_2D_PLACEHOLDER
                    amp_cdf_1d = amplitude_cdf

                ap_delay, ap_amplitude = _ap_kernel_nonuniform(
                    sel_photon_channel,
                    rU0_sel,
                    rU1,
                    delaytime_cdf,
                    amp_cdf_2d,
                    amp_cdf_1d,
                    amp_ndim == 2,
                    float(delaytime_bin_size),
                    float(amplitude_bin_size),
                    float(self.pmt_ap_t_modifier),
                )

            _photon_timings.append(merged_photon_timings[sel_photon_id] + ap_delay)
            _photon_channels.append(sel_photon_channel)
            _photon_gains.append(self._gains_f64[sel_photon_channel] * ap_amplitude)

        if not _photon_timings:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
        return (
            np.concatenate(_photon_timings),
            np.concatenate(_photon_channels).astype(np.int64),
            np.concatenate(_photon_gains),
        )


# Module-level placeholders for the kernel's unused-rank amplitude_cdf parameter.
# Numba does NOT accept a single array param whose shape (1D vs 2D) varies per
# call — so we pass both shapes always; the kernel branches on `amp_is_2d` and
# reads from whichever is real. The placeholder is a single zero element of the
# unused rank.
_AP_2D_PLACEHOLDER = np.zeros((1, 1), dtype=np.float64)
_AP_1D_PLACEHOLDER = np.zeros(1, dtype=np.float64)


@numba.njit(cache=True, nogil=True)
def _ap_select_kernel(
    rU0,
    merged_photon_channels,
    merged_photon_id_dpe,
    delaytime_cdf,
    pmt_ap_modifier,
    sel_out,
):
    """Selection pass for the per-element afterpulse scheme.

    For each row k of ``rU0``: divide by ``pmt_ap_modifier`` (and halve if
    the row is flagged as a DPE photon) in place, compare against the
    per-channel probability ceiling ``delaytime_cdf[channel, -1]``, and
    if the scaled value passes the cut, record the row index. The kernel
    also returns the maximum per-row probability ceiling for the caller's
    > 0.5 warning check.

    Modifies ``rU0`` in place. Writes the indices that pass the selection
    into ``sel_out[:n_sel]`` and returns ``(n_sel, prob_max)``. Each row's
    float ops (scalar division, halving, scalar compare) follow the same
    IEEE-754 sequence as the equivalent numpy expression.
    """
    n = rU0.shape[0]
    K_last = delaytime_cdf.shape[1] - 1
    n_sel = 0
    prob_max = -1.0
    for k in range(n):
        v = rU0[k] / pmt_ap_modifier
        if merged_photon_id_dpe[k]:
            v /= 2.0
        rU0[k] = v
        ch = merged_photon_channels[k]
        prob = delaytime_cdf[ch, K_last]
        if prob > prob_max:
            prob_max = prob
        if v <= prob:
            sel_out[n_sel] = k
            n_sel += 1
    return n_sel, prob_max


@numba.njit(cache=True, inline="always")
def _argmin_abs_diff_1d(cdf, r, K):
    """Bit-identical equivalent of ``np.argmin(np.abs(cdf - r))`` over a 1D CDF.

    Uses strict ``<`` so ties break to the leftmost index — same convention
    as ``np.argmin``. Linear O(K) scan with zero allocation; CDF bin counts
    here are ~few hundred so the scan fits in L1 and avoids the per-call
    ``(N_sel, K)`` intermediate that the broadcast-+-argmin pattern would
    materialise.
    """
    best_i = 0
    best_d = abs(cdf[0] - r)
    for i in range(1, K):
        d = abs(cdf[i] - r)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


@numba.njit(cache=True, inline="always")
def _argmin_abs_diff_row(cdf2d, row, r, K):
    """Same as ``_argmin_abs_diff_1d`` but over ``cdf2d[row, :]``."""
    best_i = 0
    best_d = abs(cdf2d[row, 0] - r)
    for i in range(1, K):
        d = abs(cdf2d[row, i] - r)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


@numba.njit(cache=True, nogil=True)
def _ap_kernel_nonuniform(
    sel_ch,
    rU0_sel,
    rU1,
    delaytime_cdf,
    amplitude_cdf_2d,
    amplitude_cdf_1d,
    amp_is_2d,
    delaytime_bin_size,
    amplitude_bin_size,
    pmt_ap_t_modifier,
):
    """Inner loop body for the non-Uniform afterpulse case.

    For each selected photon ``k`` of channel ``ch = sel_ch[k]``, computes:

      idx_d = argmin_i |delaytime_cdf[ch, i] - rU0_sel[k]|
      idx_a = argmin_i |amplitude_cdf[ch_or_flat, i] - rU1[k]|
      ap_delay[k]     = idx_d * delaytime_bin_size - pmt_ap_t_modifier
      ap_amplitude[k] = idx_a * amplitude_bin_size

    Two register-resident O(K) scans per photon (one over the delay-time
    CDF, one over the amplitude CDF). The ``amp_is_2d`` flag picks the
    active amplitude-CDF input — the unused one is a placeholder array
    (numba does not accept a single parameter whose rank varies per call,
    so both shapes are always passed).
    """
    n = sel_ch.shape[0]
    ap_delay = np.empty(n, dtype=np.float64)
    ap_amp = np.empty(n, dtype=np.float64)
    K_d = delaytime_cdf.shape[1]
    if amp_is_2d:
        K_a = amplitude_cdf_2d.shape[1]
    else:
        K_a = amplitude_cdf_1d.shape[0]

    for k in range(n):
        ch = sel_ch[k]
        idx_d = _argmin_abs_diff_row(delaytime_cdf, ch, rU0_sel[k], K_d)
        ap_delay[k] = idx_d * delaytime_bin_size - pmt_ap_t_modifier

        if amp_is_2d:
            idx_a = _argmin_abs_diff_row(amplitude_cdf_2d, ch, rU1[k], K_a)
        else:
            idx_a = _argmin_abs_diff_1d(amplitude_cdf_1d, rU1[k], K_a)
        ap_amp[k] = idx_a * amplitude_bin_size

    return ap_delay, ap_amp
