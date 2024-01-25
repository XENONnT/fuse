import strax
import numpy as np

export, __all__ = strax.exporter()

@export
class SurvivingClusters(strax.Plugin):

    __version__ = "0.0.1"

    depends_on = ("peak_basics", "contributing_clusters", "microphysics_summary")
    provides = "surviving_clusters"

    dtype = [('in_a_record', np.bool_),
             ('in_a_peak', np.bool_),
             ]
    dtype += strax.time_fields

    data_kind = "interactions_in_roi"

    def compute(self, peaks, raw_records, interactions_in_roi):

        #Check if the cluster contributes to any cluster
        cluster_in_any_record = np.isin(microphysics_summary["cluster_id"], np.unique(raw_records["contributing_clusters"]))

        contributing_clusters_per_peak = strax.split_touching_windows(raw_records, peaks)

        clusters_in_peaks = []
        for i in range(len(peaks)):
            unique_contributing_clusters = np.unique(contributing_clusters_per_peak[i]["contributing_clusters"])
            clusters_in_peaks.append(unique_contributing_clusters)
        clusters_that_make_it_into_a_peak = np.unique(np.concatenate(clusters_in_peaks))

        cluster_in_any_peak = np.isin(microphysics_summary["cluster_id"], clusters_that_make_it_into_a_peak)

        result = np.zeros(len(microphysics_summary), dtype=self.dtype)
        result['in_a_record'] = cluster_in_any_record
        result['in_a_peak'] = cluster_in_any_peak
        result['time'] = microphysics_summary['time']
        result['endtime'] = microphysics_summary['endtime']
        return result        