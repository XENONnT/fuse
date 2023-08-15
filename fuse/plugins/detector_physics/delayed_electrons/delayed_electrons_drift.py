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

    depends_on = ('photo_ionization_electrons',)
    provides = "drifted_delayed_electrons"
    data_kind = "delayed_interactions_in_roi"

    def compute(self, delayed_interactions_in_roi):
        return super().compute(interactions_in_roi=delayed_interactions_in_roi)