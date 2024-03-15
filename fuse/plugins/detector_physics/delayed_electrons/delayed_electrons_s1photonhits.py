import strax
import straxen
import logging
import numpy as np

export, __all__ = strax.exporter()

from ....plugin import FuseBasePlugin

@export
class S1PhotonHitsEmpty(FuseBasePlugin):
    """Plugin to return zeros for all S1 photon hits of delayed electrons"""

    __version__ = "0.0.1"

    depends_on = ("photo_ionization_electrons")
    provides = "delayed_s1_photons"
    data_kind = "delayed_interactions_in_roi"

    dtype = [(("Number detected S1 photons", "n_s1_photon_hits"), np.int32),
            ]
    dtype = dtype + strax.time_fields

    def compute(self,delayed_interactions_in_roi ):
        result = np.zeros(len(delayed_interactions_in_roi), dtype=self.dtype)
        result["time"] = delayed_interactions_in_roi["time"]
        result["endtime"] = delayed_interactions_in_roi["endtime"]
        return result