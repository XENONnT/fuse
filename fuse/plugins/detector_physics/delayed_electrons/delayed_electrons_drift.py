import strax
from ..electron_drift import ElectronDrift

export, __all__ = strax.exporter()

@export
class DelayedElectronsDrift(ElectronDrift):
    """
    This class is used to simulate the drift of electrons from the sources of electron afterpulses. 
    """
    __version__ = "0.0.0"
    
    child_plugin = True

    depends_on = ('delayed_electron_summary',)
    provides = "drifted_delayed_electrons"