import strax
from ..electron_timing import ElectronTiming

export, __all__ = strax.exporter()

@export
class DelayedElectronsTiming(ElectronTiming):
    """
    This class is used to simulate the timing of electrons from the sources of electron afterpulses. 
    """
    __version__ = "0.0.0"
    
    child_plugin = True

    depends_on = ('drifted_delayed_electrons','extracted_delayed_electrons')
    provides = "delayed_electron_time"