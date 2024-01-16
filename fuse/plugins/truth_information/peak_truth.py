import strax
import numpy as np

export, __all__ = strax.exporter()

@export
class PeakTruth(strax.Plugin):

    __version__ = "0.0.1"

    depends_on = ("peak_basics", "contributing_clusters", "microphysics_summary", "s1_photons", "s2_photons_sum")
    provides = "peak_truth"

    dtype = [('s1_photon_number_truth', np.int32),
             ('s2_photon_number_truth', np.int32),
             ('ap_photon_number_truth', np.int32),
             ('raw_area_truth', np.float32),
             ('observable_energy_truth', np.float32),
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

        for i in range(n_peaks):
            result['s1_photon_number_truth'][i] = contributing_clusters_per_peak[i]["s1_photons_per_cluster"].sum()
            result['s2_photon_number_truth'][i] = contributing_clusters_per_peak[i]["s2_photons_per_cluster"].sum()
            result['ap_photon_number_truth'][i] = contributing_clusters_per_peak[i]["ap_photons_per_cluster"].sum()
            result['raw_area_truth'][i] = contributing_clusters_per_peak[i]["raw_area"].sum()

            unique_contributing_clusters = np.unique(contributing_clusters_per_peak[i]["contributing_clusters"])

            result['number_of_contributing_clusters'][i] = np.sum(unique_contributing_clusters > 0)


            s1_photons_from_cluster = []
            s2_photons_from_cluster = []
            #contributing_cluster_informations = []
            for cluster_index in unique_contributing_clusters:
                if cluster_index <=0: #Skip for afterpulses and no clusters
                    continue

                s1_photons_from_cluster_tmp = np.sum(contributing_clusters_per_peak[i]["s1_photons_per_cluster"][contributing_clusters_per_peak[i]["contributing_clusters"] == cluster_index])
                s2_photons_from_cluster_tmp = np.sum(contributing_clusters_per_peak[i]["s2_photons_per_cluster"][contributing_clusters_per_peak[i]["contributing_clusters"] == cluster_index])
                
                s1_photons_from_cluster.append(s1_photons_from_cluster_tmp)
                s2_photons_from_cluster.append(s2_photons_from_cluster_tmp)

            contributing_cluster_informations = interactions_in_roi[np.isin(interactions_in_roi["cluster_id"], contributing_clusters_per_peak[i]["contributing_clusters"])]
            sort_index = np.argsort(contributing_cluster_informations["cluster_id"])
            contributing_cluster_informations = contributing_cluster_informations[sort_index]
            

            
            if len(contributing_cluster_informations)>0:

                s1_cluster_weights = np.nan_to_num(s1_photons_from_cluster/contributing_cluster_informations["n_s1_photon_hits"])
                s2_cluster_weights = np.nan_to_num(s2_photons_from_cluster/contributing_cluster_informations["sum_s2_photons"])

                result['observable_energy_truth'][i] = np.sum(contributing_cluster_informations["ed"] * s1_cluster_weights + contributing_cluster_informations["ed"] * s2_cluster_weights)
            

                physical_photons_in_peak = result['s1_photon_number_truth'][i] + result['s2_photon_number_truth'][i]
                if physical_photons_in_peak > 0:
                    result['average_x_of_contributing_clusters'][i] = np.sum(contributing_cluster_informations["x"] * s1_photons_from_cluster + contributing_cluster_informations["x"] * s2_photons_from_cluster) / physical_photons_in_peak
                    result['average_y_of_contributing_clusters'][i] = np.sum(contributing_cluster_informations["y"] * s1_photons_from_cluster + contributing_cluster_informations["y"] * s2_photons_from_cluster) / physical_photons_in_peak
                    result['average_z_of_contributing_clusters'][i] = np.sum(contributing_cluster_informations["z"] * s1_photons_from_cluster + contributing_cluster_informations["z"] * s2_photons_from_cluster) / physical_photons_in_peak
            

        return result