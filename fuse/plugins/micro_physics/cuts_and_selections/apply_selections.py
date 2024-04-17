import strax
import numpy as np

from ....plugin import FuseBasePlugin, FuseCutList
from .detector_volumes import XENONnT_TPC, XENONnT_BelowCathode
from .physics_cases import EnergyCut

export, __all__ = strax.exporter()


@export
class ApplyCuts(FuseBasePlugin):
    """Plugin that applies the previous cuts."""

    depends_on = ("clustered_interactions", "microphysics_selection")

    provides = "interactions_in_roi"
    data_kind = "interactions_in_roi"
    __version__ = "0.1.0"

    dtype = [
        (("x position of the cluster [cm]", "x"), np.float32),
        (("y position of the cluster [cm]", "y"), np.float32),
        (("z position of the cluster [cm]", "z"), np.float32),
        (("Energy of the cluster [keV]", "ed"), np.float32),
        (("NEST interaction type", "nestid"), np.int8),
        (("Mass number of the interacting particle", "A"), np.int8),
        (("Charge number of the interacting particle", "Z"), np.int8),
        (("Geant4 event ID", "evtid"), np.int32),
        (("x position of the primary particle [cm]", "x_pri"), np.float32),
        (("y position of the primary particle [cm]", "y_pri"), np.float32),
        (("z position of the primary particle [cm]", "z_pri"), np.float32),
        (("ID of the cluster", "cluster_id"), np.int32),
        (("Xenon density at the cluster position", "xe_density"), np.float32),
        (("ID of the volume in which the cluster occured", "vol_id"), np.int8),
        (("Flag indicating if a cluster can create a S2 signal", "create_S2"), np.bool_),
    ]

    dtype = dtype + strax.time_fields

    def compute(self, clustered_interactions):

        reduced_data = clustered_interactions[clustered_interactions["microphysics_selection"]]

        data = np.zeros(len(reduced_data), dtype=self.dtype)
        strax.copy_to_buffer(reduced_data, data, "_remove_cuts_and_selections")

        return data


@export
class LowEnergySimulation(FuseCutList):
    __version__ = "0.0.1"
    provides = "microphysics_selection"
    accumulated_cuts_string = "microphysics_selection"
    cuts = (XENONnT_TPC, XENONnT_BelowCathode, EnergyCut)


@export
class DefaultSimulation(FuseCutList):
    __version__ = "0.0.1"
    provides = "microphysics_selection"
    accumulated_cuts_string = "microphysics_selection"
    cuts = (XENONnT_TPC, XENONnT_BelowCathode)
