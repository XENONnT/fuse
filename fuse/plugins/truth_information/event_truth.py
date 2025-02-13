import strax
import numpy as np

export, __all__ = strax.exporter()


@export
class EventTruth(strax.Plugin):
    __version__ = "0.0.4"

    depends_on = ("microphysics_summary", "photon_summary", "peak_truth", "event_basics")
    provides = "event_truth"
    data_kind = "events"

    dtype = [
        ("x_truth", np.float32),
        ("y_truth", np.float32),
        ("z_truth", np.float32),
        ("x_obs_truth", np.float32),
        ("y_obs_truth", np.float32),
        ("z_obs_truth", np.float32),
        ("energy_of_main_peaks_truth", np.float32),
        ("total_energy_in_event_truth", np.float32),
    ] + strax.time_fields

    def compute(self, interactions_in_roi, propagated_photons, peaks, events):
        peaks_in_event = strax.split_by_containment(peaks, events)
        photons_per_event = strax.split_by_containment(propagated_photons, events)

        n_events = len(events)

        result = np.zeros(n_events, dtype=self.dtype)
        result["time"] = events["time"]
        result["endtime"] = events["endtime"]

        for i, (pic, e) in enumerate(zip(peaks_in_event, events)):
            s1 = pic[e["s1_index"]]
            s2 = pic[e["s2_index"]]

            result["x_truth"][i] = s2["average_x_of_contributing_clusters"]
            result["y_truth"][i] = s2["average_y_of_contributing_clusters"]
            result["z_truth"][i] = np.mean(
                [
                    s2["average_z_of_contributing_clusters"],
                    s1["average_z_of_contributing_clusters"],
                ]
            )

            result["x_obs_truth"][i] = s2["average_x_obs_of_contributing_clusters"]
            result["y_obs_truth"][i] = s2["average_y_obs_of_contributing_clusters"]
            result["z_obs_truth"][i] = np.mean(
                [
                    s2["average_z_obs_of_contributing_clusters"],
                    s1["average_z_obs_of_contributing_clusters"],
                ]
            )

            # Does this make any sense?
            result["energy_of_main_peaks_truth"][i] = np.mean(
                [s2["observable_energy_truth"], s1["observable_energy_truth"]]
            )

            # And lets get the total energy that is in the event
            contributing_cluster_informations = interactions_in_roi[
                np.isin(interactions_in_roi["cluster_id"], photons_per_event[i]["cluster_id"])
            ]
            # sort_index = np.argsort(contributing_cluster_informations["cluster_id"])
            # contributing_cluster_informations = contributing_cluster_informations[sort_index]
            result["total_energy_in_event_truth"][i] = np.sum(
                contributing_cluster_informations["ed"]
            )

        return result
