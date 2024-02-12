import strax
import numpy as np

export, __all__ = strax.exporter()

@export
class ClusterTagging(strax.Plugin):
    """Plugin to tag if clusters contribute to the main or alternative s1/s2"""

    __version__ = "0.0.1"

    depends_on = ("peak_basics","microphysics_summary", "event_basics", "contributing_clusters")
    provides = "tagged_clusters"
    data_kind = "interactions_in_roi"

    dtype = [('in_main_s1', np.bool_),
         ('in_main_s2', np.bool_),
         ('in_alt_s1', np.bool_),
         ('in_alt_s2', np.bool_),
        ]
    dtype = dtype + strax.time_fields   

    def compute(self, peaks, raw_records, interactions_in_roi, events):

        peaks_in_event= strax.split_by_containment(peaks, events)
        contributing_clusters_per_event = strax.split_touching_windows(raw_records, events)

        result = np.zeros(len(interactions_in_roi), dtype=self.dtype)
        result['time'] = interactions_in_roi['time']
        result['endtime'] = interactions_in_roi['endtime']

        for i, (pic, e, cie) in enumerate(zip(peaks_in_event, events, contributing_clusters_per_event)):
            s1 = pic[e['s1_index']]
            s2 = pic[e['s2_index']]

            alt_s1 = pic[e['alt_s1_index']]
            alt_s2 = pic[e['alt_s2_index']]

            clusters = strax.split_touching_windows(cie, pic)

            main_s1_clusters = clusters[e['s1_index']]
            main_s2_clusters = clusters[e['s2_index']]
            alt_s1_clusters = clusters[e['alt_s1_index']]
            alt_s2_clusters = clusters[e['alt_s2_index']]
            
            contributing_clusters_main_s1 = np.unique(main_s1_clusters['contributing_clusters'])
            contributing_clusters_main_s2 = np.unique(main_s2_clusters['contributing_clusters'])
            contributing_clusters_alt_s1 = np.unique(alt_s1_clusters['contributing_clusters'])
            contributing_clusters_alt_s2 = np.unique(alt_s2_clusters['contributing_clusters'])

            contributing_clusters_main_s1 = contributing_clusters_main_s1[contributing_clusters_main_s1>0]
            contributing_clusters_main_s2 = contributing_clusters_main_s2[contributing_clusters_main_s2>0]
            contributing_clusters_alt_s1 = contributing_clusters_alt_s1[contributing_clusters_alt_s1>0]
            contributing_clusters_alt_s2 = contributing_clusters_alt_s2[contributing_clusters_alt_s2>0]


            matching_clusters_main_s1 = np.argwhere(np.isin(interactions_in_roi["cluster_id"], contributing_clusters_main_s1))
            result["in_main_s1"][matching_clusters_main_s1] = True

            matching_clusters_main_s2 = np.argwhere(np.isin(interactions_in_roi["cluster_id"], contributing_clusters_main_s2))
            result["in_main_s2"][matching_clusters_main_s2] = True

            matching_clusters_alt_s1 = np.argwhere(np.isin(interactions_in_roi["cluster_id"], contributing_clusters_alt_s1))
            result["in_alt_s1"][matching_clusters_alt_s1] = True

            matching_clusters_alt_s2 = np.argwhere(np.isin(interactions_in_roi["cluster_id"], contributing_clusters_alt_s2))
            result["in_alt_s2"][matching_clusters_alt_s2] = True

        return result