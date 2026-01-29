import strax
from ..electron_propagation import ElectronPropagation, ElectronPropagationPerpWires

export, __all__ = strax.exporter()


@export
class DelayedElectronPropagation(ElectronPropagation):
    """This class is used to simulate the propagation of electrons from the
    sources of electron afterpulses."""

    __version__ = "0.0.0"

    child_plugin = True

    depends_on = ("photo_ionization_electrons", "drifted_delayed_electrons")
    provides = "delayed_electrons_at_interface"
    data_kind = "delayed_individual_electrons"

    def compute(self, delayed_interactions_in_roi):
        return super().compute(interactions_in_roi=delayed_interactions_in_roi)


@export
class DelayedElectronPropagationPerpWires(ElectronPropagationPerpWires):
    """This class is used to simulate the propagation of electrons from the
    sources of electron afterpulses, including the effect of perpendicular
    wires."""

    __version__ = "0.0.1"

    child_plugin = True

    depends_on = ("photo_ionization_electrons", "drifted_delayed_electrons")
    provides = "delayed_electrons_at_interface"
    data_kind = "delayed_individual_electrons"

    def compute(self, delayed_interactions_in_roi):
        return super().compute(interactions_in_roi=delayed_interactions_in_roi)
