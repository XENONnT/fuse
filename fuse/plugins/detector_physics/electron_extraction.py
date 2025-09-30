import numpy as np
import strax
import straxen

from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()


@export
class ElectronExtraction(FuseBasePlugin):
    """Plugin to simulate the loss of electrons during the extraction of
    drifted electrons from the liquid into the gas phase."""

    __version__ = "0.3.0"

    depends_on = "electrons_at_interface"
    provides = "extracted_electrons"
    data_kind = "individual_electrons"

    save_when = strax.SaveWhen.TARGET

    dtype = [
        (("x position of the electron at the interface [cm]", "x_interface"), np.float32),
        (("y position of the electron at the interface [cm]", "y_interface"), np.float32),
        (("ID of the cluster creating the electron", "cluster_id"), np.int32),
    ] + strax.time_fields

    # Config options
    s2_secondary_sc_gain_mc = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=s2_secondary_sc_gain",
        type=(int, float),
        cache=True,
        help="Secondary scintillation gain [PE/e-]",
    )
    # Rename? -> g2_value in beta_yields model
    g2_mean = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?&fmt=json&take=g2_mean",
        type=(int, float),
        cache=True,
        help="Mean value of the g2 gain [PE/e-]",
    )

    electron_extraction_yield = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=electron_extraction_yield",
        type=(int, float),
        cache=True,
        help="Electron extraction yield [electron_extracted/electron]",
    )

    ext_eff_from_map = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=ext_eff_from_map",
        type=bool,
        cache=True,
        help="Boolean indication if the extraction efficiency is taken from a map",
    )

    se_gain_from_map = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=se_gain_from_map",
        type=bool,
        cache=True,
        help="Boolean indication if the secondary scintillation gain is taken from a map",
    )

    s2_correction_map = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=s2_correction_map"
        "&fmt=json",
        cache=True,
        help="S2 correction map",
    )

    se_gain_map = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=se_gain_map"
        "&fmt=json",
        cache=True,
        help="Map of the single electron gain",
    )

    electron_trapping_time = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=electron_trapping_time",
        type=(int, float),
        cache=True,
        help="Time scale electrons are trapped at the liquid gas interface",
    )

    def compute(self, individual_electrons):

        N = individual_electrons.shape[0]

        # Fast path: scalar probability (no maps)
        if not self.ext_eff_from_map:
            p = float(self.electron_extraction_yield)
            if p <= 0.0:
                return np.zeros(0, dtype=self.dtype)
            if p >= 1.0:
                idx = np.arange(N, dtype=np.int64)
            else:
                # Faster than binomial for large N
                u = self.rng.random(N)
                idx = np.flatnonzero(u < p)
        else:
            # Only build positions if we actually need maps
            # shape: (N, 2)
            position = np.column_stack(
                (individual_electrons["x_interface"], individual_electrons["y_interface"])
            )

            # rel S2 correction (flatten for safety; maps sometimes return (N,1))
            rel_s2_cor = self.s2_correction_map(position).reshape(-1)

            if self.se_gain_from_map:
                se_gains = self.se_gain_map(position).reshape(-1)
            else:
                # keep g2 consistent with MC scaling
                se_gains = rel_s2_cor * float(self.s2_secondary_sc_gain_mc)

            # cy = g2_mean * rel_s2_cor / se_gains
            cy = (float(self.g2_mean) * rel_s2_cor) / se_gains
            # be defensive: ensure 0<=p<=1 (maps can have tiny numerical wiggles)
            np.clip(cy, 0.0, 1.0, out=cy)

            # Single RNG pass to build mask
            u = self.rng.random(N)
            idx = np.flatnonzero(u < cy)

        M = int(idx.size)
        if M == 0:
            return np.zeros(0, dtype=self.dtype)

        # Allocate result once and fill via np.take (one indexed gather per field)
        result = np.zeros(M, dtype=self.dtype)

        times_sel = np.take(individual_electrons["time"], idx)
        result["time"] = self.extraction_delay(times_sel)
        result["endtime"] = result["time"]

        result["x_interface"] = np.take(individual_electrons["x_interface"], idx)
        result["y_interface"] = np.take(individual_electrons["y_interface"], idx)
        result["cluster_id"] = np.take(individual_electrons["cluster_id"], idx)

        return result

    def extraction_delay(self, electron_times):
        # Vectorized, deterministic via self.rng
        # exponential() returns float64; cast once before adding
        dt = self.rng.exponential(float(self.electron_trapping_time), size=electron_times.shape[0])
        return electron_times + dt
