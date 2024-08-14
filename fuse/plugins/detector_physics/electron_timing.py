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

    # tf_electron_timing_near_wire_model = straxen.URLConfig(
    #     default="take://resource://"
    #     "SIMULATION_CONFIG_FILE.json?&fmt=json"
    #     "&take=electron_trapping_time",
    #     type=(int, float),
    #     cache=True,
    #     help="Time scale electrons are trapped at the liquid gas interface",
    # )
    # TODO: Add import of the TF model, can be a local file for test


    perp_wires_cut_distance = straxen.URLConfig(
        default=4.45,
        help=(
            "Distance in x to apply exception from the center of the gate perpendicular wires [cm]"
        ),
    )

    perp_wire_x_pos = straxen.URLConfig(
        default=13.06,
        help=(
            "X position of the perpendicular wires [cm]"
        ),
    )

    perp_wire_angle = straxen.URLConfig(
        default=np.deg2rad(30),
        help=(
            "Perpendicular wire angle [rad]"
        ),
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
            interactions_in_roi[mask]["x_obs"],
            interactions_in_roi[mask]["y_obs"],
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
        x_obs,
        y_obs
    ):
        time_r = np.repeat(time, n_electron.astype(np.int64))
        drift_time_mean_r = np.repeat(drift_time_mean, n_electron.astype(np.int64))
        drift_time_spread_r = np.repeat(drift_time_spread, n_electron.astype(np.int64))

        timing = np.zeros(time_r.shape[0], dtype=np.int64)
        mask_near_wires = self.perp_wire_region(x_obs, y_obs)
        # For non-wire region, use diffusion constant map
        timing[~mask_near_wires] = self.rng.exponential(self.electron_trapping_time, size=time_r.shape[0])
        timing[~mask_near_wires] += self.rng.normal(drift_time_mean_r, drift_time_spread_r, size=time_r.shape[0])
        # For wire region, use data-driven TF model
        # TODO: Add sampling from the TF model
        # 1. inputting (rot_x, rot_y, distance to the wire, drift time [ns]) to the TF model, get (time grid, PDF) as output
        # 2. Based on the PCF, convert to CDF, perform inverse 1D fit sampling, sample the electrons timing based on the number of electrons
        # 3. Assign the sampled timing to timing

        return time_r + timing.astype(np.int64)


    def perp_wire_region(self, x_obs, y_obs):
        """
        Returns a mask selecting the events near the perpendicular wires.
        """

        def rotate_axis(x_obs, y_obs):
            x_rot = np.cos(self.perp_wire_angle) * x_obs - np.sin(self.perp_wire_angle) * y_obs
            y_rot = np.sin(self.perp_wire_angle) * x_obs + np.cos(self.perp_wire_angle) * y_obs
            return x_rot, y_rot

        def get_near_wires(x_obs, y_obs, distance_cm):
            """Returns a mask selecting the events near the perpendicular wires."""
            x_rot, y_rot = rotate_axis(x_obs, y_obs)
            mask_near_wires = np.abs(np.abs(x_rot) - self.perp_wire_x_pos) < distance_cm
            return mask_near_wires

        return get_near_wires(x_obs, y_obs, self.perp_wires_cut_distance)

