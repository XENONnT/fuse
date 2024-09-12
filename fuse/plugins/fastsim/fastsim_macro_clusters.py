import strax
import numpy as np
from numba import njit
import straxen
import logging
import random

from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.fastsim.fastsim_macro_clusters")


@njit(cache=True)
def get_nn_prediction(inp, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9):
    y = np.dot(inp, w0) + w1
    y[y < 0] = 0
    y = np.dot(y, w2) + w3
    y[y < 0] = 0
    y = np.dot(y, w4) + w5
    y[y < 0] = 0
    y = np.dot(y, w6) + w7
    y[y < 0] = 0
    y = np.dot(y, w8) + w9
    y = 1 / (1 + np.exp(-y))
    return y[0]


def merge_clusters(cluster1, cluster2):
    return random.random() > 0.5


@export
class MacroClusters(FuseBasePlugin):
    """Plugin to simulate macro clusters for fastsim

    """
    __version__ = "0.0.1"

    depends_on = ("drifted_electrons", "extracted_electrons", "microphysics_summary", "s1_photon_hits")
    provides = "fastsim_macro_clusters"
    data_kind = "fastsim_macro_clusters"

    def infer_dtype(self):
        return strax.merged_dtype([self.deps[d].dtype_for(d) for d in sorted(self.depends_on)])

    def compute(self, interactions_in_roi):
        for ix1, _ in enumerate(interactions_in_roi):
            for ix2 in range(1, len(interactions_in_roi[ix1:])):
                if interactions_in_roi[ix1]['eventid'] != interactions_in_roi[ix1 + ix2]['eventid']:
                    break
                if merge_clusters(interactions_in_roi[ix1], interactions_in_roi[ix1 + ix2]):
                    ne1 = interactions_in_roi[ix1]['n_electron_extracted']
                    ne2 = interactions_in_roi[ix1 + ix2]['n_electron_extracted']
                    ne_total = ne1 + ne2
                    interactions_in_roi[ix1 + ix2]['n_electron_extracted'] = ne_total
                    interactions_in_roi[ix1]['n_electron_extracted'] = -1  # flag to throw this instruction away later

                    for quantity in ['photons', 'electrons', 'excitons', 'ed', 'n_electron_interface',
                                     'n_s1_photon_hits']:
                        interactions_in_roi[ix1 + ix2][quantity] += interactions_in_roi[ix1][quantity]

                    for quantity in ['drift_time_mean', 'drift_time_spread']:
                        interactions_in_roi[ix1 + ix2][quantity] += interactions_in_roi[ix1][quantity]
                        interactions_in_roi[ix1 + ix2][quantity] /= 2

                    if ne_total > 0:
                        for coord in ['x', 'y', 'z']:
                            for obs in ['', '_obs']:
                                interactions_in_roi[ix1 + ix2][f'{coord}{obs}'] = \
                                    (interactions_in_roi[ix1][f'{coord}{obs}'] * ne1 +
                                     interactions_in_roi[ix1 + ix2][f'{coord}{obs}'] * ne2) / ne_total

                    interactions_in_roi[ix1 + ix2]['x_pri'] = interactions_in_roi[ix1]['x_pri']
                    interactions_in_roi[ix1 + ix2]['y_pri'] = interactions_in_roi[ix1]['y_pri']
                    interactions_in_roi[ix1 + ix2]['z_pri'] = interactions_in_roi[ix1]['z_pri']

                    break
        return interactions_in_roi[interactions_in_roi['n_electron_extracted'] >= 0]
