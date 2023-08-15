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

    result_name_photons = "delayed_electrons_s2_photons"
    result_name_photons_sum = "delayed_electrons_s2_photons_sum"

    depends_on = ("drifted_delayed_electrons", "extracted_delayed_electrons", "delayed_electrons_time")

    provides = (result_name_photons, result_name_photons_sum)
    data_kind = {result_name_photons: "delayed_individual_electrons",
                 result_name_photons_sum : "delayed_interactions_in_roi"
                }
    
    dtype_photons = [('n_s2_photons', np.int64),] + strax.time_fields
    dtype_sum_photons = [('sum_s2_photons', np.int64),] + strax.time_fields
    
    dtype = dict()
    dtype[result_name_photons] = dtype_photons
    dtype[result_name_photons_sum] = dtype_sum_photons

    def compute(self, delayed_interactions_in_roi, delayed_individual_electrons):
        return super().compute(interactions_in_roi=delayed_interactions_in_roi, individual_electrons=delayed_individual_electrons)