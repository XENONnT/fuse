import strax
import numpy as np


config = straxen.get_resource('./private_nt_aux_files/sim_files/fax_config_nt_sr0_v4.json', fmt='json')


@strax.takes_config(
    strax.Option('electron_trapping_time', default=config["electron_trapping_time"], track=False, infer_type=False,
                 help="electron_trapping_time"),
)
class electron_timing(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ("drifted_electrons", "extracted_electrons")
    provides = "electron_time"
    
    data_kind = "individual_electrons"
    
    dtype = [('x', np.float64),
             ('y', np.float64),
            ]
    dtype = dtype + strax.time_fields
    #dtype = strax.time_fields
    
    def setup(self):
        pass
    
    def compute(self, electron_cloud):
        
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