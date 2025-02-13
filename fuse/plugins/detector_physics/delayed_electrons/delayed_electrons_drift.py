import strax
import straxen
from ..electron_drift import ElectronDrift

export, __all__ = strax.exporter()


@export
class DelayedElectronsDrift(ElectronDrift):
    """This class is used to simulate the drift of electrons from the sources
    of electron afterpulses."""

    __version__ = "0.0.2"

    child_plugin = True

    depends_on = "photo_ionization_electrons"
    provides = "drifted_delayed_electrons"
    data_kind = "delayed_interactions_in_roi"

    electron_lifetime_liquid_delayed_electrons = straxen.URLConfig(
        default=0,
        track=True,
        type=(int, float),
        child_option=True,
        parent_option_name="electron_lifetime_liquid",
        help="Electron lifetime in liquid xenon [ns] for delayed electrons",
    )

    def compute(self, delayed_interactions_in_roi):
        return super().compute(interactions_in_roi=delayed_interactions_in_roi)
