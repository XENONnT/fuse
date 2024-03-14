import strax
import numpy as np

export, __all__ = strax.exporter()


@export
class ClusterTagging(strax.Plugin):
    """Plugin to tag if clusters contribute to the main or alternative
    s1/s2."""

    __version__ = "0.0.2"

    depends_on = ("peak_basics", "microphysics_summary", "event_basics", "photon_summary")
    provides = "tagged_clusters"
    data_kind = "interactions_in_roi"

    dtype = [
        ("in_main_s1", np.bool_),
        ("in_main_s2", np.bool_),
        ("in_alt_s1", np.bool_),
        ("in_alt_s2", np.bool_),
        ("photons_in_main_s1", np.int32),
        ("photons_in_main_s2", np.int32),
        ("photons_in_alt_s1", np.int32),
        ("photons_in_alt_s2", np.int32),
    ]
    dtype = dtype + strax.time_fields

    def compute(self, peaks, interactions_in_roi, events, propagated_photons):
        peaks_in_event = strax.split_by_containment(peaks, events)
        photon_in_event = strax.split_touching_windows(propagated_photons, events)

        result = np.zeros(len(interactions_in_roi), dtype=self.dtype)
        result["time"] = interactions_in_roi["time"]
        result["endtime"] = interactions_in_roi["endtime"]

        for i, (peaks_of_event, event_i, photon_of_event) in enumerate(
            zip(peaks_in_event, events, photon_in_event)
        ):
            peak_photons = strax.split_touching_windows(photon_of_event, peaks_of_event)

            peak_name_dict = {
                "main_s1": "s1_index",
                "main_s2": "s2_index",
                "alt_s1": "alt_s1_index",
                "alt_s2": "alt_s2_index",
            }

            for peak_name in peak_name_dict.keys():
                peak_idx = event_i[peak_name_dict[peak_name]]
                if peak_idx >= 0:
                    photons_of_peak = peak_photons[event_i[peak_name_dict[peak_name]]]

                    contributing_clusters_of_peak, photons_in_peak = np.unique(
                        photons_of_peak["cluster_id"], return_counts=True
                    )
                    photons_in_peak = photons_in_peak[contributing_clusters_of_peak > 0]
                    contributing_clusters_of_peak = contributing_clusters_of_peak[
                        contributing_clusters_of_peak > 0
                    ]

                    matching_clusters = np.argwhere(
                        np.isin(interactions_in_roi["cluster_id"], contributing_clusters_of_peak)
                    )
                    result["in_" + peak_name][matching_clusters] = True

                    for cluster_i, photons_i in zip(contributing_clusters_of_peak, photons_in_peak):
                        mask = interactions_in_roi["cluster_id"] == cluster_i
                        result["photons_in_" + peak_name][mask] = photons_i

        return result
