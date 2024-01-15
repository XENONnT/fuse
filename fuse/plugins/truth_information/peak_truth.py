import strax
import numpy as np

export, __all__ = strax.exporter()

@export
class PeakTruth(strax.Plugin):

    __version__ = "0.0.1"

    depends_on = ("peak_basics", "contributing_clusters", "microphysics_summary")
    provides = "peak_truth"

    dtype = [('photon_number_truth', np.float32),
             ('raw_area_truth', np.float32),
             ('energy_truth', np.float32),
             ('number_of_contributing_clusters', np.int16),
             ('average_x_of_contributing_clusters', np.float32),
             ('average_y_of_contributing_clusters', np.float32),
             ('average_z_of_contributing_clusters', np.float32),
            ]
    dtype = dtype + strax.time_fields

    def compute(self, peaks, raw_records, interactions_in_roi):

        contributing_clusters_per_peak = strax.split_touching_windows(raw_records, peaks)

        n_peaks = len(peaks)

        result = np.zeros(n_peaks, dtype=self.dtype)
        result['time'] = peaks['time']
        result['endtime'] = peaks['endtime']

        photons_per_peak = np.zeros(n_peaks)
        raw_area_per_peak = np.zeros(n_peaks)
        energy_per_peak = np.zeros(n_peaks)
        number_of_contributing_clusters = np.zeros(n_peaks)
        average_x_of_contributing_clusters = np.zeros(n_peaks)
        average_y_of_contributing_clusters = np.zeros(n_peaks)
        average_z_of_contributing_clusters = np.zeros(n_peaks)
        for i in range(n_peaks):
            photons_per_peak[i] = contributing_clusters_per_peak[i]["photons_per_cluster"].sum()
            raw_area_per_peak[i] = contributing_clusters_per_peak[i]["raw_area"].sum()

            contributing_cluster_informations = interactions_in_roi[np.isin(interactions_in_roi["cluster_id"], contributing_clusters_per_peak[i]["contributing_clusters"])]
            energy_per_peak[i] = contributing_cluster_informations["ed"].sum()
            
            if contributing_cluster_informations["ed"].sum()>0:
                average_x_of_contributing_clusters[i] = np.average(contributing_cluster_informations["x"], weights = contributing_cluster_informations["ed"])
                average_y_of_contributing_clusters[i] = np.average(contributing_cluster_informations["y"], weights = contributing_cluster_informations["ed"])
                average_z_of_contributing_clusters[i] = np.average(contributing_cluster_informations["z"], weights = contributing_cluster_informations["ed"])

            number_of_contributing_clusters[i] = np.sum(np.unique(contributing_clusters_per_peak[i]["contributing_clusters"]) > 0)

        result['photon_number_truth'] = photons_per_peak
        result['raw_area_truth'] = raw_area_per_peak
        result['energy_truth'] = energy_per_peak
        result['number_of_contributing_clusters'] = number_of_contributing_clusters
        result['average_x_of_contributing_clusters'] = average_x_of_contributing_clusters
        result['average_y_of_contributing_clusters'] = average_y_of_contributing_clusters
        result['average_z_of_contributing_clusters'] = average_z_of_contributing_clusters

        return result