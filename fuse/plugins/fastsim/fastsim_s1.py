import strax
import numpy as np
import straxen
import logging

from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.fastsim.fastsim_s1")


@export
class S1Areas(FuseBasePlugin):
    """Plugin to simulate macro clusters for fastsim."""

    __version__ = "0.0.1"

    depends_on = ("fastsim_macro_clusters",)
    provides = "fastsim_s1"
    data_kind = "fastsim_events"
    dtype = [(("S1 area, uncorrected [PE]", "s1_area"), np.float32)] + strax.time_fields

    save_when = strax.SaveWhen.ALWAYS

    photon_area_distribution = straxen.URLConfig(
        default="simple_load://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=photon_area_distribution"
        "&fmt=csv",
        cache=True,
        help="Photon area distribution",
    )

    @staticmethod
    def get_s1_area_with_spe(spe_distribution, num_photons):
        """
        :params: spe_distribution, the spe distribution to draw photon areas from
        :params: num_photons, number of photons to draw from spe distribution
        """
        s1_area_spe = []
        for n_ph in num_photons:
            s1_area_spe.append(
                np.sum(
                    spe_distribution[
                        (np.random.random(n_ph) * len(spe_distribution)).astype(np.int64)
                    ]
                )
            )

        return np.array(s1_area_spe)

    def compute(self, fastsim_macro_clusters):
        eventids = np.unique(fastsim_macro_clusters["eventid"])
        result = np.zeros(len(eventids), dtype=self.dtype)
        result["time"] = fastsim_macro_clusters["time"]
        result["endtime"] = fastsim_macro_clusters["endtime"]
        for i, eventid in enumerate(eventids):
            photons = np.sum(
                fastsim_macro_clusters[fastsim_macro_clusters["eventid"] == eventid][
                    "n_s1_photon_hits"
                ]
            )
            result["s1_area"][i] = photons * 1.28
        return result
