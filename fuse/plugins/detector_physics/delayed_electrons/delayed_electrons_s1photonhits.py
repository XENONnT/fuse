import strax
import logging
import numpy as np

from ....plugin import FuseBasePlugin

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.detector_physics.delayed_electrons.delayed_electrons_s1photonhits")


@export
class S1PhotonHitsEmpty(FuseBasePlugin):
    """Plugin to return zeros for all S1 photon hits of delayed electrons."""

    __version__ = "0.0.2"

    depends_on = "photo_ionization_electrons"
    provides = "delayed_s1_photon_hits"
    data_kind = "delayed_interactions_in_roi"

    dtype = [
        (("Number detected S1 photons", "n_s1_photon_hits"), np.int32),
    ] + strax.time_fields

    def compute(self, delayed_interactions_in_roi):
        result = np.zeros(len(delayed_interactions_in_roi), dtype=self.dtype)
        result["time"] = delayed_interactions_in_roi["time"]
        result["endtime"] = delayed_interactions_in_roi["endtime"]
        return result
