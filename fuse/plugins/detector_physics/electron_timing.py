import strax
import numpy as np
import straxen

from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()


@export
class ElectronTiming(FuseBasePlugin):
    """Plugin to simulate the arrival times of electrons extracted from the
    liquid phase.

    It includes both the drift time and the time needed for the
    extraction.
    """

    __version__ = "0.2.1"

    depends_on = ("microphysics_summary", "drifted_electrons", "extracted_electrons")
    provides = "electron_time"
    data_kind = "individual_electrons"

    save_when = strax.SaveWhen.TARGET

    dtype = [
        (("x position of the electron [cm]", "x"), np.float32),
        (("y position of the electron [cm]", "y"), np.float32),
        (("ID of the cluster creating the electron", "cluster_id"), np.int32),
    ] + strax.time_fields

    # Config options
    electron_trapping_time = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=electron_trapping_time",
        type=(int, float),
        cache=True,
        help="Time scale electrons are trapped at the liquid gas interface",
    )

    def compute(self, interactions_in_roi):
        # Just apply this to clusters with photons
        mask = interactions_in_roi["n_electron_extracted"] > 0

        if len(interactions_in_roi[mask]) == 0:
            return np.zeros(0, dtype=self.dtype)

        timing = self.electron_timing(
            interactions_in_roi[mask]["time"],
            interactions_in_roi[mask]["n_electron_extracted"],
            interactions_in_roi[mask]["drift_time_mean"],
            interactions_in_roi[mask]["drift_time_spread"],
        )

        x = np.repeat(
            interactions_in_roi[mask]["x_obs"], interactions_in_roi[mask]["n_electron_extracted"]
        )
        y = np.repeat(
            interactions_in_roi[mask]["y_obs"], interactions_in_roi[mask]["n_electron_extracted"]
        )

        result = np.zeros(len(timing), dtype=self.dtype)
        result["time"] = timing
        result["endtime"] = result["time"]
        result["x"] = x
        result["y"] = y

        result["cluster_id"] = np.repeat(
            interactions_in_roi[mask]["cluster_id"],
            interactions_in_roi[mask]["n_electron_extracted"],
        )

        result = strax.sort_by_time(result)

        return result

    def electron_timing(
        self,
        time,
        n_electron,
        drift_time_mean,
        drift_time_spread,
    ):
        time_r = np.repeat(time, n_electron.astype(np.int64))
        drift_time_mean_r = np.repeat(drift_time_mean, n_electron.astype(np.int64))
        drift_time_spread_r = np.repeat(drift_time_spread, n_electron.astype(np.int64))

        timing = self.rng.exponential(self.electron_trapping_time, size=time_r.shape[0])
        timing += self.rng.normal(drift_time_mean_r, drift_time_spread_r, size=time_r.shape[0])

        return time_r + timing.astype(np.int64)
