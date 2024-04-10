import strax
import numpy as np

export, __all__ = strax.exporter()


@export
class SurvivingClusters(strax.Plugin):
    __version__ = "0.0.3"

    depends_on = ("microphysics_summary", "photon_summary", "peak_basics")
    provides = "surviving_clusters"
    data_kind = "interactions_in_roi"

    dtype = [
        ("creating_a_photon", np.bool_),
        ("in_a_peak", np.bool_),
    ]
    dtype += strax.time_fields

    def compute(self, interactions_in_roi, propagated_photons, peaks):
        # Check if the cluster contributes to any cluster
        cluster_creating_a_photon = np.isin(
            interactions_in_roi["cluster_id"], np.unique(propagated_photons["cluster_id"])
        )

        photons_per_peaks = strax.split_touching_windows(propagated_photons, peaks)

        clusters_in_peaks = []
        for i in range(len(peaks)):
            unique_contributing_clusters = np.unique(photons_per_peaks[i]["cluster_id"])
            clusters_in_peaks.append(unique_contributing_clusters)
        clusters_that_make_it_into_a_peak = np.unique(np.concatenate(clusters_in_peaks))

        cluster_in_any_peak = np.isin(
            interactions_in_roi["cluster_id"], clusters_that_make_it_into_a_peak
        )

        result = np.zeros(len(interactions_in_roi), dtype=self.dtype)
        result["creating_a_photon"] = cluster_creating_a_photon
        result["in_a_peak"] = cluster_in_any_peak
        result["time"] = interactions_in_roi["time"]
        result["endtime"] = interactions_in_roi["endtime"]
        return result
