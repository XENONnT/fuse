import strax
import numpy as np
import straxen
import logging

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.detector_physics.electron_timing')
log.setLevel('WARNING')

@export
class ElectronTiming(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ("drifted_electrons", "extracted_electrons")
    provides = "electron_time"
    data_kind = "individual_electrons"
    
    #Forbid rechunking
    rechunk_on_save = False
    

    data_kind = "individual_electrons"
    
    dtype = [('x', np.float64),
             ('y', np.float64),
            ]
    dtype = dtype + strax.time_fields
    
    #Config options
    debug = straxen.URLConfig(
        default=False, type=bool,track=False,
        help='Show debug informations',
    )

    electron_trapping_time = straxen.URLConfig(
        type=(int, float),
        help='electron_trapping_time',
    )
    
    def setup(self):
        
        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running ElectronTiming in debug mode")
    
    def compute(self, interactions_in_roi):

        #Just apply this to clusters with photons
        mask = interactions_in_roi["n_electron_extracted"] > 0
        
        if len(interactions_in_roi[mask]) == 0:
            return np.zeros(0, dtype=self.dtype)

        timing = self.electron_timing(interactions_in_roi[mask]["time"],
                                      interactions_in_roi[mask]["n_electron_extracted"],
                                      interactions_in_roi[mask]["drift_time_mean"],
                                      interactions_in_roi[mask]["drift_time_spread"]
                                     )
        
        
        x = np.repeat(interactions_in_roi[mask]["x"], interactions_in_roi[mask]["n_electron_extracted"])
        y = np.repeat(interactions_in_roi[mask]["y"], interactions_in_roi[mask]["n_electron_extracted"])
        
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