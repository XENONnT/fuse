import strax
import numpy as np
import straxen
import os
import logging

from ...common import make_map

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('XeSim.detector_physics.electron_drift')
log.setLevel('WARNING')

private_files_path = "path/to/private/files"
config = straxen.get_resource(os.path.join(private_files_path, 'sim_files/fax_config_nt_sr0_v4.json') , fmt='json')

@strax.takes_config(
    strax.Option('field_distortion_model', default=config["field_distortion_model"], track=False, infer_type=False,
                 help="field_distortion_model"),
    strax.Option('drift_velocity_liquid', default=config["drift_velocity_liquid"], track=False, infer_type=False,
                 help="drift_velocity_liquid"),
    strax.Option('field_distortion_model', default=config["field_distortion_model"], track=False, infer_type=False,
                 help="field_distortion_model"),
    strax.Option('fdc_3d',
                 default=os.path.join(private_files_path,"sim_files/XnT_3D_FDC_xyz_24_Jun_2022_MC.json.gz"),
                 track=False,
                 infer_type=False,
                 help="fdc_3d map"),
    strax.Option('field_distortion_comsol_map',
                 default=os.path.join(private_files_path,"sim_files/init_to_final_position_mapping_B2d75n_C2d75n_G0d3p_A4d9p_T0d9n_PMTs1d3n_FSR0d65p_QPTFE_0d5n_0d4p.json.gz"),
                 track=False,
                 infer_type=False,
                 help="field_distortion_comsol_map"),
    strax.Option('tpc_length', default=config["tpc_length"], track=False, infer_type=False,
                 help="tpc_length"),
    strax.Option('field_dependencies_map',
                 default=os.path.join(private_files_path,"sim_files/field_dependent_radius_depth_maps_B2d75n_C2d75n_G0d3p_A4d9p_T0d9n_PMTs1d3n_FSR0d65p_QPTFE_0d5n_0d4p.json.gz"),
                 track=False,
                 infer_type=False,
                 help="field_dependencies_map"),
    strax.Option('electron_lifetime_liquid', default=config["electron_lifetime_liquid"], track=False, infer_type=False,
                 help="electron_lifetime_liquid"),
    strax.Option('diffusion_longitudinal_map',
                 default=os.path.join(private_files_path,"sim_files/data_driven_diffusion_map_XENONnTSR0V2.json.gz"),
                 track=False,
                 infer_type=False,
                 help="diffusion_longitudinal_map"),
    strax.Option('diffusion_constant_longitudinal', default=config["diffusion_constant_longitudinal"], track=False, infer_type=False,
                 help="diffusion_constant_longitudinal"),
    strax.Option('drift_time_gate', default=config["drift_time_gate"], track=False, infer_type=False,
                 help="drift_time_gate"),
    strax.Option('debug', default=False, track=False, infer_type=False,
                 help="Show debug informations"),
)
class ElectronDrift(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ("microphysics_summary")
    provides = "drifted_electrons"
    data_kind = 'electron_cloud'
    
    dtype = [('n_electron_interface', np.int64),
             ('drift_time_mean', np.int64),
             ('drift_time_spread', np.int64),
             ('x', np.float64),
             ('y', np.float64),
             ('z_obs', np.float64),
            ]
    
    #Forbid rechunking
    rechunk_on_save = False
    
    dtype = dtype + strax.time_fields
    
    def setup(self):

        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running ElectronDrift in debug mode")
        
        if self.field_distortion_model == "inverse_fdc":
            self.fdc_3d = make_map(self.fdc_3d, fmt='json.gz')
            self.fdc_3d.scale_coordinates([1., 1., - self.drift_velocity_liquid])

        if self.field_distortion_model == "comsol":
            self.fd_comsol = make_map(self.field_distortion_comsol_map, fmt='json.gz', method='RectBivariateSpline')
            
        # Field dependencies 
        # This config entry a dictionary of 5 items
        self.enable_field_dependencies = config['enable_field_dependencies'] #This is not so nice
        if any(self.enable_field_dependencies.values()):
            field_dependencies_map_tmp = make_map(self.field_dependencies_map, fmt='json.gz', method='RectBivariateSpline')
            self.drift_velocity_scaling = 1.0
            # calculating drift velocity scaling to match total drift time for R=0 between cathode and gate
            if "norm_drift_velocity" in self.enable_field_dependencies.keys():
                if self.enable_field_dependencies['norm_drift_velocity']:
                    norm_dvel = field_dependencies_map_tmp(np.array([ [0], [- self.tpc_length]]).T, map_name='drift_speed_map')[0]
                    norm_dvel*=1e-4
                    drift_velocity_scaling = self.drift_velocity_liquid/norm_dvel
            def rz_map(z, xy, **kwargs):
                r = np.sqrt(xy[:, 0]**2 + xy[:, 1]**2)
                return field_dependencies_map_tmp(np.array([r, z]).T, **kwargs)
            self.field_dependencies_map = rz_map
            
        # Data-driven longitudinal diffusion map
        # TODO: Change to the best way to accommodate simulation/data-driven map
        if self.enable_field_dependencies["diffusion_longitudinal_map"]:
            diffusion_longitudinal_map_tmp = make_map(self.diffusion_longitudinal_map, fmt='json.gz',
                                              method='WeightedNearestNeighbors')
            def _rz_map(z, xy, **kwargs):
                r = np.sqrt(xy[:, 0]**2 + xy[:, 1]**2)
                return diffusion_longitudinal_map_tmp(np.array([r, z]).T, **kwargs)
            self.diffusion_longitudinal_map = _rz_map
    
    
    def compute(self, clustered_interactions):
        
        #Just apply this to clusters with free electrons
        instruction = clustered_interactions[clustered_interactions["electrons"] > 0]

        if len(instruction) == 0:
            return np.zeros(0, self.dtype)
        
        t = instruction["time"]
        x = instruction["x"]
        y = instruction["y"]
        z = instruction["z"]
        n_electron = instruction["electrons"].astype(np.int64)
        recoil_type = instruction["nestid"]
        recoil_type = np.where(np.isin(recoil_type, [0, 6, 7, 8, 11]), recoil_type, 8)
        
        # Reverse engineering FDC
        if self.field_distortion_model == 'inverse_fdc':
            z_obs, positions = self.inverse_field_distortion_correction(x, y, z)
        # Reverse engineering FDC
        elif self.field_distortion_model == 'comsol':
            z_obs, positions = self.field_distortion_comsol(x, y, z)
        else:
            z_obs, positions = z, np.array([x, y]).T
        
        #This logic is quite different form existing wfsim logic!
        #Remove electrons from Charge Insensitive Volume
        p_surv = self.kill_electrons(xy_int = np.array([x, y]).T, # maps are in R_true, so orginal position should be here
                                     z_int = z, # maps are in Z_true, so orginal position should be here
                                    )
        n_electron = n_electron*p_surv
        
        #Absorb electrons during the drift
        # Average drift time of the electrons
        drift_time_mean, drift_time_spread = self.get_s2_drift_time_params(xy_int = np.array([x, y]).T,
                                                                           z_int = z)
        electron_lifetime_correction = np.exp(- 1 * drift_time_mean / self.electron_lifetime_liquid)
        n_electron = n_electron*electron_lifetime_correction
        
        
        result = np.zeros(len(n_electron), dtype = self.dtype)
        result["time"] = instruction["time"]
        result["endtime"] = instruction["endtime"]
        result["n_electron_interface"] = n_electron
        result["drift_time_mean"] = drift_time_mean
        result["drift_time_spread"] = drift_time_spread
        
        #These ones are needed later
        result["x"] = positions.T[0]
        result["y"] = positions.T[1]
        result["z_obs"] = z_obs
        
        return result
        
            
            
    def inverse_field_distortion_correction(self, x, y, z):
        """For 1T the pattern map is a data driven one so we need to reverse engineer field distortion correction
        into the simulated positions
        :param x: 1d array of float
        :param y: 1d array of float
        :param z: 1d array of float
        :param resource: instance of resource class
        returns z: 1d array, postions 2d array 
        """
        positions = np.array([x, y, z]).T
        for i_iter in range(6):  # 6 iterations seems to work
            dr = self.fdc_3d(positions)
            if i_iter > 0:
                dr = 0.5 * dr + 0.5 * dr_pre  # Average between iter
            dr_pre = dr

            r_obs = np.sqrt(x**2 + y**2) - dr
            x_obs = x * r_obs / (r_obs + dr)
            y_obs = y * r_obs / (r_obs + dr)
            z_obs = - np.sqrt(z**2 + dr**2)
            positions = np.array([x_obs, y_obs, z_obs]).T

        positions = np.array([x_obs, y_obs]).T 
        return z_obs, positions
    
    def field_distortion_comsol(self, x, y, z):
        """Field distortion from the COMSOL simulation for the given electrode configuration:
        :param x: 1d array of float
        :param y: 1d array of float
        :param z: 1d array of float
        :param resource: instance of resource class
        returns z: 1d array, postions 2d array 
        """
        positions = np.array([np.sqrt(x**2 + y**2), z]).T
        theta = np.arctan2(y, x)
        r_obs = self.fd_comsol(positions, map_name='r_distortion_map')
        x_obs = r_obs * np.cos(theta)
        y_obs = r_obs * np.sin(theta)

        positions = np.array([x_obs, y_obs]).T 
        return z, positions
    
    def kill_electrons(self, xy_int, z_int):

        if self.enable_field_dependencies['survival_probability_map']:
            p_surv = self.field_dependencies_map(z_int, xy_int, map_name='survival_probability_map')
            if np.any(p_surv<0) or np.any(p_surv>1):
                # FIXME: this is necessary due to map artefacts, such as negative or values >1
                p_surv=np.clip(p_surv, a_min = 0, a_max = 1)
            
        else:
            p_surv = np.ones(len(xy_int))
        
        return p_surv
    
    
    def get_s2_drift_time_params(self, xy_int, z_int):
        """Calculate s2 drift time mean and spread
        :param z_int: 1d array of true z (floats) 
        :param xy_int: 2d array of true xy positions (floats)
        returns two arrays of floats (mean drift time, drift time spread) 
        """
        drift_velocity_liquid = self.get_avg_drift_velocity(z_int, xy_int)
        if self.enable_field_dependencies['diffusion_longitudinal_map']:
            diffusion_constant_longitudinal = self.diffusion_longitudinal_map(z_int, xy_int)  # cm²/ns
        else:
            diffusion_constant_longitudinal = self.diffusion_constant_longitudinal

        drift_time_mean = - z_int / \
            drift_velocity_liquid + self.drift_time_gate
        drift_time_mean = np.clip(drift_time_mean, 0, np.inf)
        drift_time_spread = np.sqrt(2 * self.diffusion_constant_longitudinal * drift_time_mean)
        drift_time_spread /= drift_velocity_liquid
        return drift_time_mean, drift_time_spread
    
    def get_avg_drift_velocity(self, z, xy):
        """Calculate s2 drift time mean and spread
        :param z: 1d array of z (floats)
        :param xy: 2d array of xy positions (floats)
        returns array of floats corresponding to average drift velocities from given point to the gate
        """
        if self.enable_field_dependencies['drift_speed_map']:
            drift_v_LXe = self.field_dependencies_map(z, xy, map_name='drift_speed_map')  # mm/µs
            drift_v_LXe *= 1e-4  # cm/ns
            drift_v_LXe *= self.drift_velocity_scaling
        else:
            drift_v_LXe = self.drift_velocity_liquid
        return drift_v_LXe
    

