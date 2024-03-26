import strax
import logging
from ..electron_extraction import ElectronExtraction

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.detector_physics.delayed_electrons.delayed_electrons_extraction")


@export
class DelayedElectronsExtraction(ElectronExtraction):
    """This class is used to simulate the extraction of electrons from the
    sources of electron afterpulses."""

    __version__ = "0.0.1"

    child_plugin = True

    depends_on = ("photo_ionization_electrons", "drifted_delayed_electrons")
    provides = "extracted_delayed_electrons"
    data_kind = "delayed_interactions_in_roi"

    def compute(self, delayed_interactions_in_roi):
        return super().compute(interactions_in_roi=delayed_interactions_in_roi)
