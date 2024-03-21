import strax
import numpy as np

export, __all__ = strax.exporter()


@export
class EventTruth(strax.Plugin):
    """Calculates the event truth. For observed position an
    average is taken over the contributing clusters. For the
    energy the mean is taken of s1 and s2 and the true 
    primary positions associated to this truth data is also saved.
    """

    __version__ = "0.0.2"
    
    depends_on = ("peak_truth", "microphysics_summary", "event_basics", "photon_summary")
    provides = "event_truth"
    data_kind = "events"

    dtype = [
        ("x_obs_truth", np.float32),
        ("y_obs_truth", np.float32),
        ("z_obs_truth", np.float32),
        ("x_pri", np.float32),
        ("y_pri", np.float32),
        ("z_pri", np.float32),
        ("energy_of_main_peaks_truth", np.float32),
        ("total_energy_in_event_truth", np.float32),
    ]
    dtype = dtype + strax.time_fields

    def compute(self, peaks, propagated_photons, interactions_in_roi, events):
        peaks_in_event = strax.split_by_containment(peaks, events)
        photons_per_event = strax.split_by_containment(propagated_photons, events)

        n_events = len(events)

        result = np.zeros(n_events, dtype=self.dtype)
        result["time"] = events["time"]
        result["endtime"] = events["endtime"]

        for i, (pic, e) in enumerate(zip(peaks_in_event, events)):
            s1 = pic[e["s1_index"]]
            s2 = pic[e["s2_index"]]

            result["x_obs_truth"][i] = s2["average_x_obs_of_contributing_clusters"]
            result["y_obs_truth"][i] = s2["average_y_obs_of_contributing_clusters"]
            result["z_obs_truth"][i] = np.mean(
                [
                    s2["average_z_obs_of_contributing_clusters"],
                    s1["average_z_obs_of_contributing_clusters"],
                ]
            )

            # Does this make any sense? Perhaps a weighted average or just s1?
            result["energy_of_main_peaks_truth"][i] = np.mean(
                [s2["observable_energy_truth"], s1["observable_energy_truth"]]
            )
            
            # Just the interactions that overlap with some  cluster inside our event loop
            mask = np.isin(interactions_in_roi["cluster_id"], photons_per_event[i]["cluster_id"])
            contributing_cluster_info = interactions_in_roi[mask]

            # The total energy that is in the event
            result["total_energy_in_event_truth"][i] = np.sum(contributing_cluster_info["ed"])

            # sort_index = np.argsort(contributing_cluster_info["cluster_id"])
            # contributing_cluster_info = contributing_cluster_info[sort_index]

            # Which primaries contribute to what cluster? 
            result["x_pri"][i] = np.unique(contributing_cluster_info["x_pri"])
            result["y_pri"][i] = np.unique(contributing_cluster_info["y_pri"])
            result["z_pri"][i] = np.unique(contributing_cluster_info["z_pri"])

        return result
