import strax
import numpy as np
import straxen
import logging

from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.detector_physics.electron_timing")


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

    field_dependencies_map_tmp = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=field_dependencies_map"
        "&fmt=json.gz"
        "&method=WeightedNearestNeighbors",
        cache=True,
        help="Map for the electric field dependencies",
    )

    drift_time_test = straxen.URLConfig(
        default="itp_map://resource://"
        "/home/yongyu/codes_link/fuse_examples/drift_time_test.json?&fmt=json"
        "&method=WeightedNearestNeighbors",
        cache=True,
        help="test",
    )

    drift_time_spread_test = straxen.URLConfig(
        default="itp_map://resource://"
        "/home/yongyu/codes_link/fuse_examples/drift_time_spread_test.json?&fmt=json"
        "&method=WeightedNearestNeighbors",
        cache=True,
        help="test",
    )

    drift_time_1d_perp = straxen.URLConfig(
        default="itp_map://resource://"
        "/home/yongyu/codes_link/fuse_examples/drift_time_1d_perp.json?&fmt=json"
        "&method=WeightedNearestNeighbors",
        cache=True,
        help="test",
    )

    drift_time_spread_1d_perp = straxen.URLConfig(
        default="itp_map://resource://"
        "/home/yongyu/codes_link/fuse_examples/drift_time_spread_1d_perp.json?&fmt=json"
        "&method=WeightedNearestNeighbors",
        cache=True,
        help="test",
    )

    def compute(self, interactions_in_roi):
        # Just apply this to clusters with photons
        mask = interactions_in_roi["n_electron_extracted"] > 0

        if len(interactions_in_roi[mask]) == 0:
            return np.zeros(0, dtype=self.dtype)

        def rz_map(z, xy, **kwargs):
            r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
            return self.field_dependencies_map_tmp(np.array([r, z]).T, **kwargs)

        self.field_dependencies_map = rz_map

        timing = self.electron_timing(
            interactions_in_roi[mask]["time"],
            interactions_in_roi[mask]["x"],
            interactions_in_roi[mask]["y"],
            interactions_in_roi[mask]["z"],
            interactions_in_roi[mask]["n_electron_extracted"],
            interactions_in_roi[mask]["drift_time_mean"],
            interactions_in_roi[mask]["drift_time_spread"],
            interactions_in_roi[mask]["drift_time_perp_mean"], #####
            interactions_in_roi[mask]["drift_time_perp_spread"], #####
        )

        x = np.repeat(
            interactions_in_roi[mask]["x_obs"], interactions_in_roi[mask]["n_electron_extracted"]
        )
        y = np.repeat(
            interactions_in_roi[mask]["y_obs"], interactions_in_roi[mask]["n_electron_extracted"]
        )

        # spread_test = np.full(interactions_in_roi[mask]["n_electron_extracted"], 1)

        # print(interactions_in_roi[mask]["n_electron_extracted"])
        # print(self.rng.normal(10,1,10))
        # print(spread_test)

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
        x, y, z,
        n_electron,
        drift_time_mean,
        drift_time_spread,
        drift_time_perp_mean,
        drift_time_perp_spread,
    ):
        time_r = np.repeat(time, n_electron.astype(np.int64))
        drift_time_mean_r = np.repeat(drift_time_mean, n_electron.astype(np.int64))
        drift_time_spread_r = np.repeat(drift_time_spread, n_electron.astype(np.int64))

        # drift_time_perp_mean_r = np.repeat(drift_time_perp_mean, n_electron.astype(np.int64)) ####
        # drift_time_perp_spread_r = np.repeat(drift_time_perp_spread, n_electron.astype(np.int64)) ####
        xy = np.array([x, y]).T

        diffusion_constant_radial = self.field_dependencies_map(
                z, xy, map_name="diffusion_radial_map")  # cm²/s
        diffusion_constant_azimuthal = self.field_dependencies_map(
                z, xy, map_name="diffusion_azimuthal_map")  # cm²/s
        diffusion_constant_radial *= 1e-9  # cm²/ns
        diffusion_constant_azimuthal *= 1e-9  # cm²/ns

        # print(xy, z)
        
        x_spread = np.sqrt(2 * 49*1e-9 * drift_time_mean)
        x_diff = np.concatenate([self.rng.normal(mean, std, size) 
                                for mean, std, size in zip(x, x_spread, n_electron)])
        y_diff = np.concatenate([self.rng.normal(mean, std, size) 
                                for mean, std, size in zip(y, x_spread, n_electron)])
        x_diff_rot = x_diff * np.cos(np.pi/6) - y_diff * np.sin(np.pi/6)
        y_diff_rot = x_diff * np.sin(np.pi/6) + y_diff * np.cos(np.pi/6)
        # drift_time_perp_mean_r = self.drift_time_test(np.array([x_diff_rot, y_diff_rot]).T)
        # drift_time_perp_spread_r = self.drift_time_spread_test(np.array([x_diff_rot, y_diff_rot]).T)

        x_extend = np.expand_dims(x_diff_rot, axis=1)
        drift_time_perp_mean_r = self.drift_time_1d_perp(x_extend)
        drift_time_perp_spread_r = self.drift_time_spread_1d_perp(x_extend)
        # print(self.drift_time_1d_perp([[13],[13.1]]), self.drift_time_spread_1d_perp([[13],[13.1]]))

        test_rot_xy = np.array([[-60, 60], [-30, 30], [-13.2, -13.2], [10,10], [12,12], [16, 16], [40,40]])
        # print(self.drift_time_test(test_rot_xy), self.drift_time_spread_test(test_rot_xy))

        perp_time = self.rng.normal(drift_time_perp_mean_r*1e3, drift_time_perp_spread_r*1e3, 
                                    size=time_r.shape[0])
        # print(perp_time)

        timing = self.rng.exponential(self.electron_trapping_time, size=time_r.shape[0])
        timing += self.rng.normal(drift_time_mean_r, drift_time_spread_r, size=time_r.shape[0])
        timing += perp_time

        return time_r + timing.astype(np.int64)
