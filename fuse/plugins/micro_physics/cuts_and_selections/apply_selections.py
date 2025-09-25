import strax
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

    The selection logic is given as a string expression over boolean fields
    in the `clustered_interactions` data. The expression may use '&', '|', '~',
    and parentheses. For example, to select interactions in the fiducial volume
    and with energy between 1 and 10 keV, use:
        "volume_selection & energy_range_cut"
    """

    __version__ = "1.0.0"
    save_when = strax.SaveWhen.TARGET
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
        """Tiny evaluator for '&', '|', '~', and parentheses over boolean
        fields."""
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

        mask = self._eval_logic(clustered_interactions, self.selection_logic)
        if not mask.any():
            return np.zeros(0, dtype=self.dtype)

        # Filter and project to final dtype (drops predicate fields automatically)
        reduced = clustered_interactions[mask]
        out = np.zeros((len(reduced),), dtype=self.dtype)
        strax.copy_to_buffer(reduced, out, "_copy_selected")
        return out


@export
class DefaultSimulation(SelectionMerger):
    depends_on = (
        "clustered_interactions",
        "volume_properties",
        "volume_selection",
    )
    __version__ = "1.0.2"
    selection_logic = "volume_selection"


@export
class LowEnergySimulation(SelectionMerger):
    depends_on = (
        "clustered_interactions",
        "volume_properties",
        "volume_selection",
        "energy_range_cut",
    )
    __version__ = "1.0.1"
    selection_logic = "volume_selection & energy_range_cut"
