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
            result["total_energy_in_event_truth"][i] = np.sum(
                contributing_cluster_informations["ed"]
            )

        return result


@export
class MigdalEventTruth(strax.Plugin):
    __version__ = "1.932.27"

    depends_on = ("migdal_truth", "events")
    provides = "migdal_event_truth"
    data_kind = "events"

    dtype = [
        (("Number of cluster containing a Migdal effect in event", "has_migdal"), np.uint8),
        (
            ("Number of photons at interaction position caused by Migdal effect", "migdal_photons"),
            np.int32,
        ),
        (
            (
                "Number of electrons at interaction position caused by Migdal effect",
                "migdal_electrons",
            ),
            np.int32,
        ),
        (
            (
                "Number of excitons at interaction position caused by Migdal effect",
                "migdal_excitons",
            ),
            np.int32,
        ),
        (("Total deposited ER energy", "migdal_deposited_energy"), np.float32),
        (
            (
                "Orbital of Migdal electron (first digit, n; second digit, l; sign, s=±1/2. "
                "Shows only orbital of most energetic electron, if multiple",
                "migdal_orbital",
            ),
            np.int16,
        ),
    ] + strax.time_fields

    def compute(self, interactions_in_roi, events):
        interactions_in_event = strax.split_by_containment(interactions_in_roi, events)
        n_events = len(events)

        result = np.zeros(n_events, dtype=self.dtype)
        result["time"] = events["time"]
        result["endtime"] = events["endtime"]

        for event, interactions in enumerate(interactions_in_event):
            result["has_migdal"][event] = np.sum(interactions["has_migdal"])
            result["migdal_electrons"][event] = np.sum(interactions["migdal_electrons"])
            result["migdal_photons"][event] = np.sum(interactions["migdal_photons"])
            result["migdal_excitons"][event] = np.sum(interactions["migdal_excitons"])
            result["migdal_deposited_energy"][event] = np.sum(
                interactions["migdal_deposited_energy"]
            )

            max_migdal_energy_index = np.argmax(interactions["migdal_deposited_energy"])
            result["migdal_orbital"][event] = interactions[max_migdal_energy_index][
                "migdal_orbital"
            ]

        return result
