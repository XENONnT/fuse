import strax
import numpy as np

from ....plugin import FuseBasePlugin

from ....dtypes import (
    primary_positions_fields,
    cluster_positions_fields,
    cluster_id_fields,
    cluster_misc_fields,
)

export, __all__ = strax.exporter()


@export
class SelectionMerger(FuseBasePlugin):
    """Base class for cut and selection merger plugins."""

    __version__ = "0.0.1"

    save_when = strax.SaveWhen.TARGET

    provides = "interactions_in_roi"
    data_kind = "interactions_in_roi"

    dtype = (
        cluster_positions_fields
        + cluster_id_fields
        + cluster_misc_fields
        + primary_positions_fields
        + strax.time_fields
    )

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

        for volume in self.volume_names:
            reduced_data["create_S2"] = np.where(
                reduced_data[f"{volume}_selection"],
                self.create_s2_dict[volume],
                reduced_data["create_S2"],
            )

            reduced_data["xe_density"] = np.where(
                reduced_data[f"{volume}_selection"],
                self.density_dict[volume],
                reduced_data["xe_density"],
            )

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
