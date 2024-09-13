import strax
from ..electron_extraction import ElectronExtraction

export, __all__ = strax.exporter()


@export
class DelayedElectronsExtraction(ElectronExtraction):
    """This class is used to simulate the extraction of electrons from the
    sources of electron afterpulses."""

    __version__ = "0.0.2"

    child_plugin = True

    depends_on = "delayed_electrons_at_interface"
    provides = "extracted_delayed_electrons"
    data_kind = "delayed_individual_electrons"

    def compute(self, delayed_individual_electrons):
        return super().compute(individual_electrons=delayed_individual_electrons)
