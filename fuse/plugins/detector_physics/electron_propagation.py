import strax
import straxen
import numpy as np
from numba import njit

from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()

@export
class ElectronPropagation(FuseBasePlugin):
    """Plugin to simulate the propagation of electrons in the TPC to the gas interface"""

    __version__ = "0.0.0"

    depends_on = ("drifted_electrons", "microphysics_summary")
    provides = "electrons_at_interface"
    data_kind = "individual_electrons"

    save_when = strax.SaveWhen.TARGET #?

    dtype = [
        (("x position of the electron at the interface [cm]", "x_interface"), np.float32),
        (("y position of the electron at the interface [cm]", "y_interface"), np.float32),
        (("ID of the cluster creating the electron", "cluster_id"), np.int32),
    ] + strax.time_fields

    diffusion_constant_transverse = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=diffusion_constant_transverse",
        type=(int, float),
        cache=True,
        help="Transverse diffusion constant [cm^2/ns]",
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

    def setup(self):
        super().setup()

        # Field dependencies
        if self.enable_diffusion_transverse_map:

            def rz_map(z, xy, **kwargs):
                r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
                return self.field_dependencies_map_tmp(np.array([r, z]).T, **kwargs)

            self.field_dependencies_map = rz_map

    def compute(self, interactions_in_roi):
        # Just apply this to clusters with electrons reaching the interface
        mask = interactions_in_roi["n_electron_interface"] > 0

        if len(interactions_in_roi[mask]) == 0:
            return np.zeros(0, dtype=self.dtype)
        
        electron_drift_time = drift_time_in_tpc(
            interactions_in_roi[mask]["n_electron_interface"],
            interactions_in_roi[mask]["drift_time_mean"],
            interactions_in_roi[mask]["drift_time_spread"],
            self.rng,
        )

        positions = np.array(
            [interactions_in_roi[mask]["x_obs"], interactions_in_roi[mask]["y_obs"]]
        ).T

        if self.enable_diffusion_transverse_map:
            diffusion_constant_radial = self.field_dependencies_map(
                interactions_in_roi[mask]["z_obs"], positions, map_name="diffusion_radial_map"
            )  # cm²/s
            diffusion_constant_azimuthal = self.field_dependencies_map(
                interactions_in_roi[mask]["z_obs"], positions, map_name="diffusion_azimuthal_map"
            )  # cm²/s
            diffusion_constant_radial *= 1e-9  # cm²/ns
            diffusion_constant_azimuthal *= 1e-9  # cm²/ns
        else:
            diffusion_constant_radial = self.diffusion_constant_transverse
            diffusion_constant_azimuthal = self.diffusion_constant_transverse

        hdiff = np.zeros((electron_drift_time.shape[0], 2))

        simulate_horizontal_shift(
            interactions_in_roi[mask]["n_electron_interface"],
            electron_drift_time,
            positions,
            diffusion_constant_radial,
            diffusion_constant_azimuthal,
            hdiff,
            self.rng,
        )

        positions_shifted = np.repeat(
            positions, interactions_in_roi[mask]["n_electron_interface"], axis=0
        ) + hdiff

        # Now we have the positions of the electrons at the top of the LXe
        # Simulation of wire effects go in here -> time shift + position shift


        cluster_id = np.repeat(
            interactions_in_roi[mask]["cluster_id"], interactions_in_roi[mask]["n_electron_interface"], axis=0
        )

        electron_times = np.repeat(
            interactions_in_roi[mask]["time"], interactions_in_roi[mask]["n_electron_interface"], axis=0
        ) + electron_drift_time 

        result = np.zeros(electron_drift_time.shape[0], dtype=self.dtype)
        result["time"] = electron_times
        result["endtime"] = result["time"]
        result["x_interface"] = positions_shifted[:,0]
        result["y_interface"] = positions_shifted[:,1]
        result["cluster_id"] = cluster_id

        return result


def drift_time_in_tpc(
    n_electron,
    drift_time_mean,
    drift_time_spread,
    rng
):
    n_electrons = np.sum(n_electron).astype(np.int64)
    drift_time_mean_r = np.repeat(drift_time_mean, n_electron.astype(np.int64))
    drift_time_spread_r = np.repeat(drift_time_spread, n_electron.astype(np.int64))

    timing = rng.normal(drift_time_mean_r, drift_time_spread_r, size=n_electrons)

    return timing.astype(np.int64)

@njit()
def simulate_horizontal_shift(
    n_electron,
    drift_time_electron,
    xy,
    diffusion_constant_radial,
    diffusion_constant_azimuthal,
    result,
    rng,
):
    hdiff_stdev_radial = np.sqrt(2 * diffusion_constant_radial * drift_time_electron)
    hdiff_stdev_azimuthal = np.sqrt(2 * diffusion_constant_azimuthal * drift_time_electron)

    hdiff_radial = rng.normal(0, 1, np.sum(n_electron)) * hdiff_stdev_radial
    hdiff_azimuthal = rng.normal(0, 1, np.sum(n_electron)) * hdiff_stdev_azimuthal

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