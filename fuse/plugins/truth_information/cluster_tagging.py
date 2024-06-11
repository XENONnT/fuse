import strax
import straxen
import numpy as np

export, __all__ = strax.exporter()


@export
class ClusterTagging(strax.Plugin):
    """Plugin to tag if clusters contribute to the main or alternative s1/s2 in
    an event, or successfully reconstructed as s0/s1/s2 peaks."""

    __version__ = "0.0.5"

    depends_on = ("microphysics_summary", "photon_summary", "peak_basics", "event_basics")
    provides = "tagged_clusters"
    data_kind = "interactions_in_roi"

    dtype = [
        # Tags by events
        (("Cluster in the main S1 of an event", "in_main_s1"), np.bool_),
        (("Cluster in the main S2 of an event", "in_main_s2"), np.bool_),
        (("Cluster in the alternative S1 of an event", "in_alt_s1"), np.bool_),
        (("Cluster in the alternative S2 of an event", "in_alt_s2"), np.bool_),
        (("Number of photons in the main S1", "photons_in_main_s1"), np.int32),
        (("Number of photons in the main S2", "photons_in_main_s2"), np.int32),
        (("Number of photons in the alternative S1", "photons_in_alt_s1"), np.int32),
        (("Number of photons in the alternative S2", "photons_in_alt_s2"), np.int32),
        # Tags by S1 peaks
        (("Cluster results in an S1", "has_s1"), np.bool_),
        (("S1 Channel within tight range of mean", "s1_tight_coincidence"), np.int32),
    ] + strax.time_fields

    photon_finding_window = straxen.URLConfig(
        default=200,
        type=int,
        help="Time window [ns] that defines whether a photon is in a peak. "
        "Peaks' start and end times are extended by this window to find photons in them.",
    )

    def compute(self, interactions_in_roi, propagated_photons, peaks, events):
        result = np.zeros(len(interactions_in_roi), dtype=self.dtype)
        result["time"] = interactions_in_roi["time"]
        result["endtime"] = interactions_in_roi["endtime"]

        # First we tag the clusters that are in a peak
        s1_peaks = peaks[peaks["type"] == 1]
        photons_in_peak = strax.split_touching_windows(
            interactions_in_roi, s1_peaks, window=self.photon_finding_window
        )
        for peak, photons in zip(s1_peaks, photons_in_peak):
            mask = np.isin(interactions_in_roi["cluster_id"], photons["cluster_id"])
            result["has_s1"][mask] = True
            result["s1_tight_coincidence"][mask] = peak["tight_coincidence"]

        # Then we tag the clusters that are in an event's main or alternative s1/s2
        peaks_in_event = strax.split_by_containment(peaks, events)
        photon_in_event = strax.split_touching_windows(propagated_photons, events)

        for i, (peaks_of_event, event_i, photon_of_event) in enumerate(
            zip(peaks_in_event, events, photon_in_event)
        ):
            peak_photons = strax.split_touching_windows(
                photon_of_event, peaks_of_event, window=self.photon_finding_window
            )

            peak_name_dict = {
                "main_s1": "s1_index",
                "main_s2": "s2_index",
                "alt_s1": "alt_s1_index",
                "alt_s2": "alt_s2_index",
            }

            for peak_name_cluster, peak_name_event in peak_name_dict.items():
                peak_idx = event_i[peak_name_event]
                if peak_idx >= 0:
                    photons_of_peak = peak_photons[event_i[peak_name_event]]

                    mask = photons_of_peak["cluster_id"] > 0
                    contributing_clusters_of_peak, photons_in_peak = np.unique(
                        photons_of_peak["cluster_id"][mask], return_counts=True
                    )

                    matching_clusters = np.argwhere(
                        np.isin(interactions_in_roi["cluster_id"], contributing_clusters_of_peak)
                    )
                    result["in_" + peak_name_cluster][matching_clusters] = True

                    for cluster_i, photons_i in zip(contributing_clusters_of_peak, photons_in_peak):
                        mask = interactions_in_roi["cluster_id"] == cluster_i
                        result["photons_in_" + peak_name_cluster][mask] = photons_i

        return result
