import strax
import straxen
import numpy as np

from ....plugin import FuseBasePlugin

from ....dtypes import (
    primary_positions_fields,
    cluster_positions_fields,
    cluster_id_fields,
    cluster_misc_fields,
    volume_properties_fields,
)

export, __all__ = strax.exporter()

@export
class SelectionMerger(FuseBasePlugin):
    """Merge cuts/selections and stamp per-volume constants.

    Subclasses set:
      - LOGIC: boolean expression in terms of field names (e.g. ("AND", "tpc_selection", "energy_range_cut"))
    """

    __version__ = "0.7.0"
    save_when = strax.SaveWhen.TARGET
    depensds_on = ("clustered_interactions",)
    provides = "interactions_in_roi"
    data_kind = "interactions_in_roi"

    selection_logic = "volume_selection"

    dtype = (
        cluster_positions_fields
        + cluster_id_fields
        + cluster_misc_fields
        + primary_positions_fields
        + volume_properties_fields
        + strax.time_fields
    )

    @staticmethod
    def _eval_logic(arr, expr: str) -> np.ndarray:
        """Tiny evaluator for '&', '|', '~', and parentheses over boolean fields."""
        # Build an environment from available boolean-likes.
        # (We expose ALL fields; the expression must reference valid names.)
        env = {name: arr[name].astype(bool, copy=False) for name in arr.dtype.names}
        # Safe eval: strip builtins.
        out = eval(expr, {"__builtins__": None}, env)
        # Normalize to a 1D boolean numpy array
        return np.asarray(out, dtype=np.bool_)

    def compute(self, clustered_interactions):
        if len(clustered_interactions) == 0:
            return np.zeros(0, dtype=self.dtype)

        self.log.debug(f"Applying selection logic: {self.selection_logic}")

        mask = self._eval_logic(clustered_interactions, self.selection_logic)
        if not mask.any():
            return np.zeros(0, dtype=self.dtype)

        # Filter and project to final dtype (drops predicate fields automatically)
        reduced = clustered_interactions[mask]
        out = np.empty(len(reduced), dtype=self.dtype)
        strax.copy_to_buffer(reduced, out, "_copy_selected")
        return out

@export
class LowEnergySimulation(SelectionMerger):
    __version__ = "0.8.0"
    depends_on = (
        "clustered_interactions",
        "volume_selection",
        "energy_range_cut",
    )

    selection_logic = "volume_selection & energy_range_cut"


@export
class DefaultSimulation(SelectionMerger):
    __version__ = "0.9.0"
    depends_on = (
        "clustered_interactions",
        "volume_selection",
    )

    selection_logic = "volume_selection"