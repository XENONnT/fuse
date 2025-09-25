import strax
import logging
import numpy as np

from ....common import VOLUMES_IDS

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.micro_physics.detector_volumes")


class VolumeSelection(strax.CutPlugin):
    """Plugin that evaluates if interactions are in a defined detector
    volume."""

    depends_on = "volume_properties"
    provides = "volume_selection"
    __version__ = "0.0.1"

    accept_volumes = ["tpc", "below_cathode"]

    def cut_by(self, clustered_interactions):

        mask = np.zeros(len(clustered_interactions), dtype=bool)

        accept_volume_ids = [VOLUMES_IDS[v] for v in self.accept_volumes]
        for v_id in accept_volume_ids:
            mask |= clustered_interactions["vol_id"] == v_id

        return mask
