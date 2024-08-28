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


    tf_electron_timing_near_wire_model = straxen.URLConfig(
        default="tf://"
        "/project2/lgrandi/yongyu/analysiscode/S2_shape_width/near_wire_model/Data_driven/"
        "s2_width_wired_model_temp.tar.gz",
        cache=True,
        help="test",
    )

    model_train_mean = straxen.URLConfig(
        default="take://resource://"
        "/project2/lgrandi/yongyu/data/data_v14/tf_model_input/x_train_mean_v1.npy?&fmt=npy",
        cache=True,
        help="test",
    )

    model_train_std = straxen.URLConfig(
        default="take://resource://"
        "/project2/lgrandi/yongyu/data/data_v14/tf_model_input/x_train_std_v1.npy?&fmt=npy",
        cache=True,
        help="test",
    )

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
        time_grid = np.linspace(-30,30,300) # [us]

        rot_x, rot_y = self.rotate_axis(x_obs, y_obs)
        diff_wire = self.distance_to_wire(x_obs, y_obs)
        
        # Input for ML model
        time_r = np.repeat(time, n_electron.astype(np.int64))
        x_obs_r = np.repeat(x_obs, n_electron.astype(np.int64))
        y_obs_r = np.repeat(y_obs, n_electron.astype(np.int64))
        x_rot_r = np.repeat(rot_x, n_electron.astype(np.int64))
        y_rot_r = np.repeat(rot_y, n_electron.astype(np.int64))
        diff_wire_r = np.repeat(diff_wire, n_electron.astype(np.int64))

        drift_time_mean_r = np.repeat(drift_time_mean, n_electron.astype(np.int64))
        drift_time_spread_r = np.repeat(drift_time_spread, n_electron.astype(np.int64))

        timing = np.zeros(time_r.shape[0], dtype=np.float64)
        mask_near_wires = self.perp_wire_region(x_obs_r, y_obs_r)
        
        # For non-wire region, use diffusion constant map
        timing[~mask_near_wires] = self.rng.exponential(self.electron_trapping_time, 
                                                        size=time_r[~mask_near_wires].shape[0])
        timing[~mask_near_wires] += self.rng.normal(drift_time_mean_r[~mask_near_wires], 
                                                    drift_time_spread_r[~mask_near_wires], 
                                                    size=time_r[~mask_near_wires].shape[0])
        
        # For wire region, use data-driven TF model
        drift_time_temp = self.electron_trapping_time + drift_time_mean
        drift_time_temp_r = np.repeat(drift_time_temp, n_electron.astype(np.int64))

        self.model = self.tf_electron_timing_near_wire_model
        input_model = np.column_stack((x_obs_r[mask_near_wires], 
                                      y_obs_r[mask_near_wires], 
                                      x_rot_r[mask_near_wires], 
                                      y_rot_r[mask_near_wires], 
                                      diff_wire_r[mask_near_wires], 
                                      drift_time_temp_r[mask_near_wires]))
        input_model = (input_model - self.model_train_mean) / self.model_train_std

        pdf = self.model(input_model)
        cdf = np.cumsum(pdf, axis=1)
        cdf = cdf / cdf[:, -1].reshape(-1, 1)  # normalization
        timing_temp = np.array([np.interp(np.random.rand(), cdf[i], time_grid) 
                                for i in range(cdf.shape[0])])
        timing_temp *= 1e3 # convert us to ns
        timing[mask_near_wires] = drift_time_temp_r[mask_near_wires] + timing_temp

        return time_r + timing.astype(np.int64)

    def rotate_axis(self, x_obs, y_obs):
        x_rot = np.cos(self.perp_wire_angle) * x_obs - np.sin(self.perp_wire_angle) * y_obs
        y_rot = np.sin(self.perp_wire_angle) * x_obs + np.cos(self.perp_wire_angle) * y_obs
        return x_rot, y_rot

    def distance_to_wire(self, x_obs, y_obs):
        x_rot, y_rot = self.rotate_axis(x_obs, y_obs)
        return np.abs(x_rot) - self.perp_wire_x_pos
    
    def perp_wire_region(self, x_obs, y_obs):
        distance_cm = 5
        mask_near_wires = np.abs(self.distance_to_wire(x_obs, y_obs)) < distance_cm
        return mask_near_wires

