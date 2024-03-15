import strax
import logging
from ..electron_timing import ElectronTiming

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.detector_physics.delayed_electrons.delayed_electrons_timing")


@export
class DelayedElectronsTiming(ElectronTiming):
    """This class is used to simulate the timing of electrons from the sources
    of electron afterpulses."""

    __version__ = "0.0.1"

    child_plugin = True

    depends_on = (
        "drifted_delayed_electrons",
        "extracted_delayed_electrons",
        "photo_ionization_electrons",
    )
    provides = "delayed_electrons_time"
    data_kind = "delayed_individual_electrons"

    def compute(self, delayed_interactions_in_roi):
        return super().compute(interactions_in_roi=delayed_interactions_in_roi)
