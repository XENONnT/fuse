import strax
import numpy as np
import straxen
import os
import logging

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('XeSim.detector_physics.electron_timing')
log.setLevel('WARNING')

private_files_path = "path/to/private/files"
config = straxen.get_resource(os.path.join(private_files_path, 'sim_files/fax_config_nt_sr0_v4.json') , fmt='json')


@strax.takes_config(
    strax.Option('electron_trapping_time', default=config["electron_trapping_time"], track=False, infer_type=False,
                 help="electron_trapping_time"),
    strax.Option('debug', default=False, track=False, infer_type=False,
                 help="Show debug informations"),
)
class ElectronTiming(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ("drifted_electrons", "extracted_electrons")
    provides = "electron_time"
    
    #Forbid rechunking
    rechunk_on_save = False
    

    data_kind = "individual_electrons"
    
    dtype = [('x', np.float64),
             ('y', np.float64),
            ]
    dtype = dtype + strax.time_fields
    #dtype = strax.time_fields
    
    def setup(self):
        
        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running ElectronTiming in debug mode")
    
    def compute(self, electron_cloud):
        
        if len(electron_cloud) == 0:
            return np.zeros(0, dtype=self.dtype)

        timing = self.electron_timing(electron_cloud["time"],
                                      electron_cloud["n_electron_extracted"],
                                      electron_cloud["drift_time_mean"],
                                      electron_cloud["drift_time_spread"]
                                     )
        
        
        x = np.repeat(electron_cloud["x"], electron_cloud["n_electron_extracted"])
        y = np.repeat(electron_cloud["y"], electron_cloud["n_electron_extracted"])
        
        result = np.zeros(len(timing), dtype = self.dtype)
        result["time"] = timing
        result["endtime"] = result["time"]
        result["x"] = x
        result["y"] = y
        
        return result
        
    
    def electron_timing(self,
                        time,
                        n_electron,
                        drift_time_mean,
                        drift_time_spread,
                        ):

        time_r = np.repeat(time, n_electron.astype(np.int64))
        drift_time_mean_r = np.repeat(drift_time_mean, n_electron.astype(np.int64))
        drift_time_spread_r = np.repeat(drift_time_spread, n_electron.astype(np.int64))

        timing = np.random.exponential(self.electron_trapping_time, size = time_r.shape[0])
        timing += np.random.normal(drift_time_mean_r, drift_time_spread_r, size = time_r.shape[0])

        return time_r + timing.astype(np.int64)