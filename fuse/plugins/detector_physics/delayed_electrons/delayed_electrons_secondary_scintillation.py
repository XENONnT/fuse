import strax
import numpy as np
from ..secondary_scintillation import SecondaryScintillation

export, __all__ = strax.exporter()

@export
class DelayedElectronsSecondaryScintillation(SecondaryScintillation):
    """
    This class is used to simulate the extraction of electrons from the sources of electron afterpulses. 
    """
    __version__ = "0.0.0"
    
    child_plugin = True

    depends_on = ("drifted_delayed_electrons", "extracted_delayed_electrons", "delayed_electron_time")

    provides = ("delayed_electrons_s2_photons", "delayed_electrons_s2_photons_sum")
    data_kind = {"delayed_electrons_s2_photons": "individual_electrons",
                 "delayed_electrons_s2_photons_sum" : "interactions_in_roi"
                }
    
    dtype_photons = [('n_s2_photons', np.int64),] + strax.time_fields
    dtype_sum_photons = [('sum_s2_photons', np.int64),] + strax.time_fields
    
    dtype = dict()
    dtype["delayed_electrons_s2_photons"] = dtype_photons
    dtype["delayed_electrons_s2_photons_sum"] = dtype_sum_photons