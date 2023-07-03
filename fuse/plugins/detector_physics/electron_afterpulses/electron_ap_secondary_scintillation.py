import strax
import numpy as np
from ..secondary_scintillation import SecondaryScintillation

export, __all__ = strax.exporter()

@export
class ElectronAfterpulsesSecondaryScintillation(SecondaryScintillation):
    """
    This class is used to simulate the extraction of electrons from the sources of electron afterpulses. 
    """
    __version__ = "0.0.0"
    
    child_plugin = True

    depends_on = ("drifted_electrons","extracted_electrons" ,"electron_time")

    provides = ("ap_s2_photons", "ap_s2_photons_sum")
    data_kind = {"ap_s2_photons": "individual_electrons",
                 "ap_s2_photons_sum" : "interactions_in_roi"
                }
    
    dtype_photons = [('n_s2_photons', np.int64),] + strax.time_fields
    dtype_sum_photons = [('sum_s2_photons', np.int64),] + strax.time_fields
    
    dtype = dict()
    dtype["ap_s2_photons"] = dtype_photons
    dtype["ap_s2_photons_sum"] = dtype_sum_photons