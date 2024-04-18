import strax
import numpy as np

from ....plugin import FuseBasePlugin
from .detector_volumes import XENONnT_TPC, XENONnT_BelowCathode
from .physics_cases import EnergyCut

export, __all__ = strax.exporter()


@export
class SelectionMerger(FuseBasePlugin):
    """Base class for cut and selection merger plugins."""

    __version__ = "0.0.1"

    save_when = strax.SaveWhen.TARGET

    provides = "interactions_in_roi"
    data_kind = "interactions_in_roi"
    __version__ = "0.0.1"

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

    def setup(self):
        # I can get the conditions in the volumes from the plugin lineage!
        self.selections_the_plugin_depends_on = [p for p in self.depends_on if "_selection" in p]
        self.cuts_the_plugin_depends_on = [p for p in self.depends_on if "_cut" in p]
        self.cuts_and_selections = (
            self.selections_the_plugin_depends_on + self.cuts_the_plugin_depends_on
        )
        self.volume_names = [p[:-10] for p in self.selections_the_plugin_depends_on]

        self.density_dict = {}
        self.create_s2_dict = {}
        for volume in self.volume_names:
            self.density_dict[volume] = self.lineage[f"{volume}_selection"][2][
                f"xenon_density_{volume}"
            ]
            self.create_s2_dict[volume] = self.lineage[f"{volume}_selection"][2][
                f"create_S2_xenonnt_{volume}"
            ]

    def compute(self, clustered_interactions):

        combined_selection = get_accumulated_bool(clustered_interactions[self.cuts_and_selections])

        reduced_data = clustered_interactions[combined_selection]

        data = np.zeros(len(reduced_data), dtype=self.dtype)
        strax.copy_to_buffer(reduced_data, data, "_remove_cuts_and_selections")

        return data


def get_accumulated_bool(array):
    """Computes accumulated boolean over all cuts and selections.

    :param array: Array containing merged cuts.
    """
    fields = array.dtype.names
    fields = np.array([f for f in fields if f not in ("time", "endtime")])

    res = np.zeros(len(array), np.bool_)
    # Modified from the default code
    for field in fields:
        if field.endswith("_selection"):
            res |= array[field]

    for field in fields:
        if field.endswith("_cut"):
            res &= array[field]

    return res


@export
class LowEnergySimulation(SelectionMerger):
    __version__ = "0.0.1"
    depends_on = [
        "clustered_interactions",
        "tpc_selection",
        "below_cathode_selection",
        "energy_range_cut",
    ]


@export
class DefaultSimulation(SelectionMerger):
    __version__ = "0.0.2"
    depends_on = ("clustered_interactions", "tpc_selection", "below_cathode_selection")
