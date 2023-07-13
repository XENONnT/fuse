import strax
from ..electron_extraction import ElectronExtraction

export, __all__ = strax.exporter()

@export
class DelayedElectronsExtraction(ElectronExtraction):
    """
    This class is used to simulate the extraction of electrons from the sources of electron afterpulses. 
    """
    __version__ = "0.0.0"
    
    child_plugin = True

    depends_on = ('delayed_electron_summary','drifted_delayed_electrons')
    provides = "extracted_delayed_electrons"