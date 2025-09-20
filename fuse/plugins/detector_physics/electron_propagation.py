import strax
import straxen
import numpy as np
from numba import njit, prange

from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()


# =========================
# Numba helpers (no RNG)
# =========================

@njit(cache=True)
def _build_ranges(n_electron):
    """Return start/stop indices per interaction and total electrons."""
    nint = n_electron.shape[0]
    start_idx = np.empty(nint, np.int64)
    stop_idx  = np.empty(nint, np.int64)
    total = 0
    for i in range(nint):
        k = np.int64(n_electron[i])
        start_idx[i] = total
        total += k
        stop_idx[i] = total
    return start_idx, stop_idx, total  # total == N_electrons


@njit(parallel=True, fastmath=True, cache=True)
def rotate_radial_azimuthal_to_xy_inplace(
    n_electron_per_int,   # (N_int,) int64
    theta_per_int,        # (N_int,) float32/64
    hr,                   # (N_elec,) float32/64   radial kicks
    ha,                   # (N_elec,) float32/64   azimuthal kicks
    out_dxdy              # (N_elec,2) float32/64  preallocated
):
    """Fast block-rotation (hr, ha) -> (dx, dy), per interaction angle."""
    start_idx, stop_idx, _ = _build_ranges(n_electron_per_int)
    nint = n_electron_per_int.shape[0]

    for k in prange(nint):
        s = start_idx[k]
        e = stop_idx[k]
        th = theta_per_int[k]
        c  = np.cos(th)
        sn = np.sin(th)
        for j in range(s, e):
            r = hr[j]
            a = ha[j]
            # (radial, azimuthal) -> (x, y)
            out_dxdy[j, 0] = r * c + a * sn
            out_dxdy[j, 1] = -r * sn + a * c


# =========================
# Plugins
# =========================

@export
class ElectronPropagation(FuseBasePlugin):
    """Plugin to simulate the propagation of electrons in the TPC to the gas
    interface."""

    __version__ = "0.0.3"  # bumped due to performance refactor

    depends_on = ("drifted_electrons", "microphysics_summary")
    provides = "electrons_at_interface"
    data_kind = "individual_electrons"

    save_when = strax.SaveWhen.TARGET

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

        # Per-interaction electron counts
        n_int = interactions_in_roi[mask]["n_electron_interface"].astype(np.int64)

        # -----------------------------
        # Drift time per electron (ns)
        # Use your deterministic RNG from FuseBasePlugin
        # -----------------------------
        electron_drift_time = drift_time_in_tpc(
            n_int,
            interactions_in_roi[mask]["drift_time_mean"],
            interactions_in_roi[mask]["drift_time_spread"],
            self.rng,
        ).astype(np.float32)
        Ne = electron_drift_time.shape[0]

        # Per-interaction positions and angles
        positions = np.column_stack(
            (interactions_in_roi[mask]["x_obs"], interactions_in_roi[mask]["y_obs"])
        ).astype(np.float32)
        theta = np.arctan2(positions[:, 1], positions[:, 0]).astype(np.float32)

        # -----------------------------
        # Diffusion constants (cm^2/ns), expand to per-electron
        # -----------------------------
        if self.enable_diffusion_transverse_map:
            diffusion_constant_radial_int = self.field_dependencies_map(
                interactions_in_roi[mask]["z_obs"], positions, map_name="diffusion_radial_map"
            ).astype(np.float32) * 1e-9  # s -> ns
            diffusion_constant_azimuthal_int = self.field_dependencies_map(
                interactions_in_roi[mask]["z_obs"], positions, map_name="diffusion_azimuthal_map"
            ).astype(np.float32) * 1e-9
        else:
            # treat as scalars
            diffusion_constant_radial_int = np.array(
                [self.diffusion_constant_transverse], dtype=np.float32
            )
            diffusion_constant_azimuthal_int = np.array(
                [self.diffusion_constant_transverse], dtype=np.float32
            )

        if diffusion_constant_radial_int.size == 1:
            D_r_e = np.full(Ne, diffusion_constant_radial_int[0], dtype=np.float32)
            D_a_e = np.full(Ne, diffusion_constant_azimuthal_int[0], dtype=np.float32)
        else:
            D_r_e = np.repeat(diffusion_constant_radial_int, n_int).astype(np.float32)
            D_a_e = np.repeat(diffusion_constant_azimuthal_int, n_int).astype(np.float32)

        # -----------------------------
        # Horizontal diffusion kicks (deterministic: from self.rng)
        # std = sqrt(2 * D * t)
        # -----------------------------
        std_r = np.sqrt(2.0 * D_r_e * electron_drift_time).astype(np.float32)
        std_a = np.sqrt(2.0 * D_a_e * electron_drift_time).astype(np.float32)

        z_r = self.rng.normal(0.0, 1.0, size=Ne).astype(np.float32)
        z_a = self.rng.normal(0.0, 1.0, size=Ne).astype(np.float32)
        hr = z_r * std_r
        ha = z_a * std_a

        # Rotate to xy with Numba (fast)
        hdiff = np.empty((Ne, 2), dtype=np.float32)
        rotate_radial_azimuthal_to_xy_inplace(
            n_int,
            theta,
            hr,
            ha,
            hdiff,
        )

        # Apply shift
        positions_shifted = np.repeat(positions, n_int, axis=0)
        positions_shifted += hdiff

        # Now we have the positions of the electrons at the top of the LXe
        # Times at interface (ns)
        electron_times = (
            np.repeat(
                interactions_in_roi[mask]["time"],
                n_int,
                axis=0,
            ).astype(np.int64)
            + electron_drift_time.astype(np.int64)
        )

        # Simulation of wire effects -> time shift + position shift
        positions_shifted, electron_times = self.apply_perpendicular_wire_effects(
            positions_shifted, electron_times
        )

        cluster_id = np.repeat(
            interactions_in_roi[mask]["cluster_id"],
            n_int,
            axis=0,
        )

        result = np.zeros(Ne, dtype=self.dtype)
        result["time"] = electron_times
        result["endtime"] = result["time"]
        result["x_interface"] = positions_shifted[:, 0]
        result["y_interface"] = positions_shifted[:, 1]
        result["cluster_id"] = cluster_id

        return result

    def apply_perpendicular_wire_effects(self, positions, times):
        """Apply the time and position shift due to the perpendicular wires."""
        return positions, times


@export
class ElectronPropagationPerpWires(ElectronPropagation):
    """Plugin to simulate the propagation of electrons in the TPC to the gas
    interface, including the effect of the perpendicular wires on time and
    position.
    
    - We use two 1D maps for the position shift, one for the left side of the
      perpendicular wires and one for the right side. The need for two maps is to
      have a sharp transition at the perpendicular wire position, that the interpolation
      would otherwise smooth out. The position maps are just an offset, so we
      add the offset to the x position in the rotated frame.

    - We use two 1D maps for the time shift, one for the mean and one for the spread.
      The time shift is applied as a random number drawn from a Gaussian with
      mean and spread given by the maps.
    """

    __version__ = "0.0.2"  # bumped to reflect refactor in base

    enable_perp_wire_electron_shift = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=enable_perp_wire_electron_shift",
        type=bool,
        cache=True,
        help="Enable the time and position shift due to the perpendicular wires",
    )

    perp_wires_x_position_offset_1d_mean_left = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=perp_wires_x_position_offset_1d_mean_left"
        "&fmt=json"
        "&method=WeightedNearestNeighbors",
        cache=True,
        help="test",
    )

    perp_wires_x_position_offset_1d_mean_right = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=perp_wires_x_position_offset_1d_mean_right"
        "&fmt=json"
        "&method=WeightedNearestNeighbors",
        cache=True,
        help="test",
    )

    perp_wires_drift_time_1d = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=perp_wires_drift_time_1d"
        "&fmt=json"
        "&method=WeightedNearestNeighbors",
        cache=True,
        help="test",
    )

    perp_wires_drift_time_spread_1d = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=perp_wires_drift_time_spread_1d"
        "&fmt=json"
        "&method=WeightedNearestNeighbors",
        cache=True,
        help="test",
    )

    perp_wire_x_pos = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=perp_wire_x_pos",
        type=(int, float),
        cache=True,
        help="X position of the perpendicular wires [cm]",
    )

    perp_wire_angle = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=perp_wire_angle",
        type=(int, float),
        cache=True,
        help="Angle of the perpendicular wires [deg]",
    )

    perp_wires_rot_x_mask = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=perp_wires_rot_x_mask",
        type=(list, tuple),
        cache=True,
        help="Distance in x (cm) from the center of the gate perpendicular wires; "
        "expects a list or tuple [left, right]",
    )

    position_correction_pp_wire_shift = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=position_correction_pp_wire_shift",
        type=(int, float),
        cache=True,
        help="Distance in x (cm) from the center of the gate perpendicular wires "
        "to apply different position corrections",
    )

    def setup(self):
        super().setup()
        self.perp_wire_angle_rad = np.deg2rad(self.perp_wire_angle)

    def apply_perpendicular_wire_effects(self, positions, times):
        """Apply the time and position shift due to the perpendicular wires."""
        if self.enable_perp_wire_electron_shift:
            self.log.debug("Applying perpendicular wire effects")
            times = self.time_correction_pp_wire(times, positions)
            positions = self.position_correction_pp_wire(positions)
        return positions, times

    def position_correction_pp_wire(self, positions):
        """
        Apply the position shift due to the perpendicular wires.
        The map is defined in the rotated x frame, so we need to rotate the
        positions to apply the correction, and then rotate back.
        We pass the absolute value of x to the map, and then apply the
        correction with the appropriate sign.
        """
        x_rot, y_rot = rotate_axis(positions[:, 0], positions[:, 1], self.perp_wire_angle_rad)

        x_diff = np.zeros(positions.shape[0], dtype=positions.dtype)
        # Get a mask close to wires
        # it selects two regions, one with positive x_rot and one with negative x_rot
        mask_near_wires = self.get_near_wires_mask(positions)
        # We split the mask in two, one for the left side of the wire and one for the right side
        mask_near_wires_left = mask_near_wires & (
            np.abs(x_rot) < self.position_correction_pp_wire_shift
        )
        mask_near_wires_right = mask_near_wires & (
            np.abs(x_rot) >= self.position_correction_pp_wire_shift
        )

        x_rot = np.expand_dims(x_rot, axis=1)
        x_rot_left = x_rot[mask_near_wires_left]
        x_rot_right = x_rot[mask_near_wires_right]

        # Apply position correction only to electrons close to the wires
        # passing the absolute value of x_rot to the maps
        x_diff[mask_near_wires_left] = self.perp_wires_x_position_offset_1d_mean_left(
            np.abs(x_rot_left)
        )
        x_diff[mask_near_wires_right] = self.perp_wires_x_position_offset_1d_mean_right(
            np.abs(x_rot_right)
        )

        x_diff = np.expand_dims(x_diff, axis=1)
        # Add the offset to the x position in the rotated frame
        x_rot_shifted = x_rot + x_diff

        # Inverse rotation to get back to the original frame
        x_obs_shifted, y_obs_shifted = rotate_axis(
            x_rot_shifted.flatten(), y_rot, -self.perp_wire_angle_rad
        )
        positions = np.column_stack([x_obs_shifted, y_obs_shifted])

        return positions

    def time_correction_pp_wire(self, time, positions):
        """
        Current behavior: add only the mean time shift from the map (in us),
        converted to ns. If you later want Gaussian smearing, sample with
        self.rng.normal(mean, spread) here (deterministic).
        """
        x_rot, _ = rotate_axis(positions[:, 0], positions[:, 1], self.perp_wire_angle_rad)

        x_extend = np.expand_dims(x_rot, axis=1)
        drift_time_perp_mean_r = self.perp_wires_drift_time_1d(x_extend)  # in us
        # drift_time_perp_spread_r = self.perp_wires_drift_time_spread_1d(x_extend)  # in us

        # Deterministic: just add the mean (ns)
        perp_time = drift_time_perp_mean_r.flatten().astype(np.float32) * 1e3  # ns
        return time + perp_time

    def get_near_wires_mask(self, positions):
        """Returns a mask selecting the events near the perpendicular wires."""
        x_rot, _ = rotate_axis(positions[:, 0], positions[:, 1], self.perp_wire_angle_rad)
        mask_near_wires = np.abs(x_rot) - self.perp_wire_x_pos < self.perp_wires_rot_x_mask[1]
        mask_near_wires &= np.abs(x_rot) - self.perp_wire_x_pos > -self.perp_wires_rot_x_mask[0]
        return mask_near_wires


# =========================
# Helpers
# =========================

def drift_time_in_tpc(n_electron, drift_time_mean, drift_time_spread, rng):
    """Draw per-electron drift times with the plugin's deterministic RNG."""
    n_electrons = np.sum(n_electron).astype(np.int64)
    drift_time_mean_r = np.repeat(drift_time_mean, n_electron.astype(np.int64))
    drift_time_spread_r = np.repeat(drift_time_spread, n_electron.astype(np.int64))
    timing = rng.normal(drift_time_mean_r, drift_time_spread_r, size=n_electrons)
    return timing.astype(np.int64)


def rotate_axis(x_obs, y_obs, angle):
    x_rot = np.cos(angle) * x_obs - np.sin(angle) * y_obs
    y_rot = np.sin(angle) * x_obs + np.cos(angle) * y_obs
    return x_rot, y_rot