import strax
import numpy as np
from ..s2_photon_propagation import S2PhotonPropagation

export, __all__ = strax.exporter()

@export
class ElectronAfterpulsesS2PhotonPropagation(S2PhotonPropagation):
    """
    Add some documentation here
    """
    __version__ = "0.0.0"
    
    child_plugin = True

    depends_on = ("ap_s2_photons", "extracted_ap_electrons", "drifted_ap_electrons", "ap_s2_photons_sum")
    provides = "propagated_ap_s2_photons"