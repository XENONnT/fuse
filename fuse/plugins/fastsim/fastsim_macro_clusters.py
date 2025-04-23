import inspect
import os
import pickle

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


@export
class MacroClusters(FuseBasePlugin):
    """Plugin to simulate macro clusters for fastsim."""

    __version__ = "0.0.1"

    depends_on = (
        "drifted_electrons",
        "extracted_electrons",
        "microphysics_summary",
        "s1_photon_hits",
    )
    provides = "fastsim_macro_clusters"
    data_kind = "fastsim_macro_clusters"

    electron_trapping_time = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=electron_trapping_time",
        type=(int, float),
        cache=True,
        help="Time scale electrons are trapped at the liquid gas interface",
    )

    p_double_pe_emission = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=p_double_pe_emision",
        type=(int, float),
        cache=True,
        help="Probability of double photo-electron emission",
    )

    def infer_dtype(self):
        return strax.merged_dtype([self.deps[d].dtype_for(d) for d in sorted(self.depends_on)])

    def compute(self, interactions_in_roi):
        for ix1, _ in enumerate(interactions_in_roi):
            for ix2 in range(1, len(interactions_in_roi[ix1:])):
                if interactions_in_roi[ix1]["eventid"] != interactions_in_roi[ix1 + ix2]["eventid"]:
                    break
                if self.merge_these_clusters(
                    interactions_in_roi[ix1], interactions_in_roi[ix1 + ix2]
                ):
                    ne1 = interactions_in_roi[ix1]["n_electron_extracted"]
                    ne2 = interactions_in_roi[ix1 + ix2]["n_electron_extracted"]
                    ne_total = ne1 + ne2
                    interactions_in_roi[ix1 + ix2]["n_electron_extracted"] = ne_total
                    interactions_in_roi[ix1]["n_electron_extracted"] = -1

                for q in [
                    "photons",
                    "electrons",
                    "excitons",
                    "ed",
                    "n_electron_interface",
                    "n_s1_photon_hits",
                ]:
                    interactions_in_roi[ix1 + ix2][q] += interactions_in_roi[ix1][q]
                for q in ["drift_time_mean", "drift_time_spread"]:
                    interactions_in_roi[ix1 + ix2][q] += interactions_in_roi[ix1][q]
                    interactions_in_roi[ix1 + ix2][q] /= 2

                if ne_total > 0:
                    for coord in ["x", "y", "z"]:
                        for obs in ["", "_obs"]:
                            interactions_in_roi[ix1 + ix2][f"{coord}{obs}"] = (
                                interactions_in_roi[ix1][f"{coord}{obs}"] * ne1
                                + interactions_in_roi[ix1 + ix2][f"{coord}{obs}"] * ne2
                            ) / ne_total

                interactions_in_roi[ix1 + ix2]["x_pri"] = interactions_in_roi[ix1]["x_pri"]
                interactions_in_roi[ix1 + ix2]["y_pri"] = interactions_in_roi[ix1]["y_pri"]
                interactions_in_roi[ix1 + ix2]["z_pri"] = interactions_in_roi[ix1]["z_pri"]

                break

        return interactions_in_roi[interactions_in_roi["sum_s2_photons"] >= 0]

    def merge_these_clusters(self, cluster_a, cluster_b):
        area = np.zeros(2)
        dt_mean = np.zeros(2)
        width = np.zeros(2)
        for i, inst in enumerate([cluster_a, cluster_b]):
            area[i] = inst["n_electron_extracted"] * (1 + self.p_double_pe_emission)
            dt_mean[i] = inst["drift_time_mean"]
            width[i] = np.sqrt(
                inst["drift_time_spread"] ** 2 + (self.electron_trapping_time + 78.3) ** 2
            )  # 1 sigma spread
            # TODO: s2_time_spread=78.3 was part of fax config but has been removed since. Probably need to re-add it?
        alt, main = area.argsort()
        # TODO : Get normalisation factors and NN weights from private_nt_aux_files
        sum_area = np.sum(area) / 229551
        main_width = width[main] / 5871
        alt_width = width[alt] / 5871
        delta_t = (dt_mean[main] - dt_mean[alt]) / 22378
        working_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        w0, w1, w2, w3, w4, w5, w6, w7, w8, w9 = pickle.load(
            open(f"{working_dir}/nn_weights.p", "rb+")
        )

        X = np.array((sum_area, main_width, alt_width, delta_t), dtype=np.float32)
        y = get_nn_prediction(X, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9)
        return y > 0.5
