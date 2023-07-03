import strax
from ..electron_timing import ElectronTiming

export, __all__ = strax.exporter()

@export
class ElectronAfterpulsesTiming(ElectronTiming):
    """
    This class is used to simulate the timing of electrons from the sources of electron afterpulses. 
    """
    __version__ = "0.0.0"
    
    child_plugin = True

    depends_on = ('drifted_ap_electrons','extracted_ap_electrons')
    provides = "ap_electron_time"