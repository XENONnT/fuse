import strax
import straxen
import numpy as np

from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()


@export
class ElectronDrift(FuseBasePlugin):
    """Plugin to simulate the drift of electrons from the interaction site to
    the liquid gas interface.

    The plugin simulates the effect of a charge insensitive volume and
    the loss of electrons due to impurities. Additionally, the drift
    time and observed position is calculated.
    """

    __version__ = "0.3.0"

    depends_on = "microphysics_summary"
    provides = "drifted_electrons"
    data_kind = "interactions_in_roi"

    dtype = [
        (
            ("Number of electrons reaching the liquid gas interface", "n_electron_interface"),
            np.int32,
        ),
        (("Mean drift time of the electrons in the cluster [ns]", "drift_time_mean"), np.int32),
        (
            ("Spread of the drift time of the electrons in the cluster [ns]", "drift_time_spread"),
            np.int32,
        ),
        (("Observed x position of the cluster at liquid-gas interface [cm]", "x_obs"), np.float32),
        (("Observed y position of the cluster at liquid-gas interface [cm]", "y_obs"), np.float32),
        (
            ("Observed z position of the cluster after field distortion correction [cm]", "z_obs"),
            np.float32,
        ),
    ] + strax.time_fields

    save_when = strax.SaveWhen.ALWAYS

    # Config options

    drift_velocity_liquid = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=drift_velocity_liquid",
        type=(int, float),
        cache=True,
        help="Drift velocity of electrons in the liquid xenon [cm/ns]",
    )

    drift_time_gate = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=drift_time_gate",
        type=(int, float),
        cache=True,
        help="Electron drift time from the gate [ns]",
    )

    diffusion_constant_longitudinal = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=diffusion_constant_longitudinal",
        type=(int, float),
        cache=True,
        help="Longitudinal electron drift diffusion constant [cm^2/ns]",
    )

    electron_lifetime_liquid = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=electron_lifetime_liquid",
        type=(int, float),
        cache=True,
        help="Electron lifetime in liquid xenon [ns]",
    )

    enable_survival_probability_map = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=enable_survival_probability_map",
        type=bool,
        cache=True,
        help="Use survival probability map from field_dependencies_map_tmp",
    )

    enable_drift_velocity_map = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=enable_drift_velocity_map",
        type=bool,
        cache=True,
        help="Use drift velocity map from field_dependencies_map_tmp",
    )

    norm_drift_velocity = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=norm_drift_velocity",
        type=bool,
        cache=True,
        help="Normalize drift velocity map according to value defined by drift_velocity_liquid",
    )

    enable_diffusion_longitudinal_map = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=enable_diffusion_longitudinal_map",
        type=bool,
        cache=True,
        help="Use diffusion longitudinal map from field_dependencies_map_tmp",
    )

    tpc_length = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?&fmt=json&take=tpc_length",
        type=(int, float),
        cache=True,
        help="Length of the XENONnT TPC [cm]",
    )

    field_distortion_model = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=field_distortion_model",
        cache=True,
        help="Model for the electric field distortion",
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

    diffusion_longitudinal_map_tmp = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=diffusion_longitudinal_map"
        "&fmt=json.gz"
        "&method=WeightedNearestNeighbors",
        cache=True,
        help="Longitudinal diffusion map",
    )

    fdc_map_fuse = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=field_distortion_comsol_map"
        "&fmt=json.gz"
        "&method=WeightedNearestNeighbors",
        cache=True,
        help="Field distortion map used in fuse (Check if we can remove _fuse from the name)",
    )

    gate_to_anode_distance = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?"
        "&fmt=json"
        "&take=gate_to_anode_distance",
        cache=True,
        help="Distance between the liquid surface and anode in cm",
    )

    elr_gas_gap_length = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?"
        "&fmt=json"
        "&take=elr_gas_gap_length",
        cache=True,
        help="Distance between the gate and anode in cm",
    )

    def setup(self):
        super().setup()

        # Can this be done via URL config?
        if self.field_distortion_model == "inverse_fdc":
            self.fdc_map_fuse.scale_coordinates([1.0, 1.0, -self.drift_velocity_liquid])

        # Field dependencies
        if self.enable_drift_velocity_map:
            self.drift_velocity_scaling = 1.0
            # Calculating drift velocity scaling to match total drift time
            # for R = 0 between cathode and gate
            if self.norm_drift_velocity:
                norm_dvel = self.field_dependencies_map_tmp(
                    np.array([[0], [-self.tpc_length]]).T, map_name="drift_speed_map"
                )[0]
                norm_dvel *= 1e-4
                self.drift_velocity_scaling = self.drift_velocity_liquid / norm_dvel

            def rz_map(z, xy, **kwargs):
                r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
                return self.field_dependencies_map_tmp(np.array([r, z]).T, **kwargs)

            self.field_dependencies_map = rz_map

        # Data-driven longitudinal diffusion map
        # TODO: Change to the best way to accommodate simulation/data-driven map
        if self.enable_diffusion_longitudinal_map:

            def _rz_map(z, xy, **kwargs):
                r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
                return self.diffusion_longitudinal_map_tmp(np.array([r, z]).T, **kwargs)

            self.diffusion_longitudinal_map = _rz_map

    def compute(self, interactions_in_roi):
        # Just apply this to clusters with photons
        mask = interactions_in_roi["electrons"] > 0

        if len(interactions_in_roi[mask]) == 0:
            empty_result = np.zeros(len(interactions_in_roi), self.dtype)
            empty_result["time"] = interactions_in_roi["time"]
            empty_result["endtime"] = interactions_in_roi["endtime"]
            return empty_result

        x = interactions_in_roi[mask]["x"]
        y = interactions_in_roi[mask]["y"]
        z = interactions_in_roi[mask]["z"]
        n_electron = interactions_in_roi[mask]["electrons"].astype(np.int64)

        # Reverse engineering FDC
        if self.field_distortion_model == "inverse_fdc":
            z_obs, positions = self.inverse_field_distortion_correction(x, y, z)
        # Reverse engineering FDC
        elif self.field_distortion_model == "comsol":
            z_obs, positions = self.field_distortion_comsol(x, y, z)
        else:
            z_obs, positions = z, np.array([x, y]).T

        # Remove electrons from Charge Insensitive Volume
        # interaction_in_civ can either be 0 or 1
        interaction_in_civ = self.in_charge_sensitive_volume(
            xy_int=np.array([x, y]).T,  # maps are in R_true, so orginal position should be here
            z_int=z,  # maps are in Z_true, so orginal position should be here
        )
        n_electron = self.rng.binomial(n=n_electron, p=interaction_in_civ)

        # Absorb electrons during the drift
        # Average drift time of the electrons
        drift_time_mean, drift_time_spread = self.get_s2_drift_time_params(
            xy_int=np.array([x, y]).T, z_int=z
        )

        if self.electron_lifetime_liquid > 0:
            electron_lifetime_correction = np.exp(
                -1 * drift_time_mean / self.electron_lifetime_liquid
            )

            n_electron = self.rng.binomial(
                n=n_electron.astype(np.int32), p=electron_lifetime_correction
            )
        else:
            self.log.debug("No electron lifetime applied")

        result = np.zeros(len(interactions_in_roi), dtype=self.dtype)
        result["time"] = interactions_in_roi["time"]
        result["endtime"] = interactions_in_roi["endtime"]
        result["n_electron_interface"][mask] = n_electron
        result["drift_time_mean"][mask] = drift_time_mean
        result["drift_time_spread"][mask] = drift_time_spread

        # These ones are needed later
        result["x_obs"][mask] = positions.T[0]
        result["y_obs"][mask] = positions.T[1]
        result["z_obs"][mask] = z_obs

        return result

    def inverse_field_distortion_correction(self, x, y, z):
        """For 1T the pattern map is a data driven one so we need to
        reverse engineer field distortion correction into the simulated positions
        Args:
            x: 1d array of float
            y: 1d array of float
            z: 1d array of float
            resource: instance of resource class
        Returns:
            z: 1d array, postions 2d array
        """
        positions = np.array([x, y, z]).T
        dr_pre = self.fdc_map_fuse(positions)
        for i_iter in range(6):  # 6 iterations seems to work
            dr = 0.5 * self.fdc_map_fuse(positions) + 0.5 * dr_pre  # Average between iter

            r_obs = np.sqrt(x**2 + y**2) - dr
            x_obs = x * r_obs / (r_obs + dr)
            y_obs = y * r_obs / (r_obs + dr)
            z_obs = -np.sqrt(z**2 + dr**2)
            positions = np.array([x_obs, y_obs, z_obs]).T

            dr_pre = dr

        positions = np.array([x_obs, y_obs]).T
        return z_obs, positions

    def field_distortion_comsol(self, x, y, z):
        """Field distortion from the COMSOL simulation for the given electrode configuration:
        Args:
            x: 1d array of float
            y: 1d array of float
            z: 1d array of float
            resource: instance of resource class
        Returns:
            z: 1d array, postions 2d array
        """
        positions = np.array([np.sqrt(x**2 + y**2), z]).T
        theta = np.arctan2(y, x)
        r_obs = self.fdc_map_fuse(positions, map_name="r_distortion_map")
        x_obs = r_obs * np.cos(theta)
        y_obs = r_obs * np.sin(theta)

        positions = np.array([x_obs, y_obs]).T
        return z, positions

    def in_charge_sensitive_volume(self, xy_int, z_int):
        if self.enable_survival_probability_map:
            p_surv = self.field_dependencies_map(z_int, xy_int, map_name="survival_probability_map")
            p_surv = np.clip(p_surv, a_min=0, a_max=1)

        else:
            p_surv = np.ones(len(xy_int))

        return p_surv

    def get_s2_drift_time_params(self, xy_int, z_int):
        """Calculate s2 drift time mean and spread.

        Args:
            z_int: 1d array of true z (floats)
            xy_int: 2d array of true xy positions (floats)
        Returns:
            returns two arrays of floats (mean drift time, drift time spread)
        """
        liquid_level = self.gate_to_anode_distance - self.elr_gas_gap_length
        drift_velocity_below_gate = self.get_avg_drift_velocity(z_int, xy_int)
        drift_velocity_above_gate = liquid_level / self.drift_time_gate
        if self.enable_diffusion_longitudinal_map:
            diffusion_constant_longitudinal = self.diffusion_longitudinal_map(
                z_int, xy_int
            )  # cm²/ns
        else:
            diffusion_constant_longitudinal = self.diffusion_constant_longitudinal

        drift_time_below_gate = -z_int / drift_velocity_below_gate
        drift_time_above_gate = self.drift_time_gate

        drift_time_mean = drift_time_below_gate + drift_time_above_gate
        drift_time_mean = np.clip(drift_time_mean, 0, np.inf)

        drift_time_spread_below_gate_squared = (
            2
            * diffusion_constant_longitudinal
            * drift_time_below_gate
            / drift_velocity_below_gate**2
        )
        drift_time_spread_above_gate_squared = (
            2
            * diffusion_constant_longitudinal
            * drift_time_above_gate
            / drift_velocity_above_gate**2
        )
        drift_time_spread = np.sqrt(
            drift_time_spread_below_gate_squared + drift_time_spread_above_gate_squared
        )

        return drift_time_mean, drift_time_spread

    def get_avg_drift_velocity(self, z, xy):
        """Calculate s2 drift time mean and spread
        Args:
            z: 1d array of z (floats)
            xy: 2d array of xy positions (floats)
        Returns:
            array of floats corresponding to average drift velocities
                from given point to the gate.
        """
        if self.enable_drift_velocity_map:
            drift_v_LXe = self.field_dependencies_map(z, xy, map_name="drift_speed_map")  # mm/µs
            drift_v_LXe *= 1e-4  # cm/ns
            drift_v_LXe *= self.drift_velocity_scaling
        else:
            drift_v_LXe = self.drift_velocity_liquid
        return drift_v_LXe
