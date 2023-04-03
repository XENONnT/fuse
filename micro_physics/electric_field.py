import strax
import epix
import numpy as np

@strax.takes_config(
    strax.Option('debug', default=False, track=False, infer_type=False,
                 help="Show debug informations"),
    strax.Option('Detector', default="XENONnT", track=False, infer_type=False,
                 help="Detector to be used. Has to be defined in epix.detectors"),
    strax.Option('DetectorConfigOverride', default=None, track=False, infer_type=False,
                 help="Config file to overwrite default epix.detectors settings; see examples in the configs folder"),
)
class electic_field(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ("clustered_interactions",)
    provides = "electic_field_values"
    
    dtype = [('e_field', np.int64),
            ]
    
    dtype = dtype + strax.time_fields
    
    def setup(self):
        #Do the volume cuts here #Maybe we can move these lines somewhere else?
        self.detector_config = epix.init_detector(self.Detector.lower(), self.DetectorConfigOverride)
 
    #Why is geant4_interactions given from straxen? It should be called clustered_interactions or??
    def compute(self, geant4_interactions):
        
        result = np.zeros(len(geant4_interactions), dtype=self.dtype)
        result["time"] = geant4_interactions["time"]
        result["endtime"] = geant4_interactions["endtime"]

        efields = result["e_field"]
        
        for volume in self.detector_config:
            if isinstance(volume.electric_field, (float, int)):
                ids = geant4_interactions['vol_id']
                m = ids == volume.volume_id
                efields[m] = volume.electric_field
            else:
                efields = volume.electric_field(geant4_interactions.x,
                                                geant4_interactions.y,
                                                geant4_interactions.z
                                                )

        result["e_field"] = efields
        
        return result
        
        