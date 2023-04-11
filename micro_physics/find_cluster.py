import numpy as np
import pandas as pd
import numba
import strax
import epix
import awkward as ak
from epix.common import reshape_awkward

from sklearn.cluster import DBSCAN

@strax.takes_config(
    strax.Option('micro_separation', default=0.005, track=False, infer_type=False,
                 help="DBSCAN clustering distance (mm)"),
    strax.Option('micro_separation_time', default = 10, track=False, infer_type=False,
                 help="Clustering time (ns)"),
    strax.Option('debug', default=False, track=False, infer_type=False,
                 help="Show debug informations"),
)
class find_cluster(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ("geant4_interactions",)
    
    provides = "cluster_index"
    
    dtype = [('cluster_ids', np.int64),
            ]
    dtype = dtype + strax.time_fields

    #Forbid rechunking
    rechunk_on_save = False
    

    def compute(self, geant4_interactions):
        
        inter = ak.from_numpy(np.empty(1, dtype=geant4_interactions.dtype))
        structure = geant4_interactions["structure"][geant4_interactions["structure"]>=0]
        
        for field in inter.fields:
            inter[field] = epix.reshape_awkward(geant4_interactions[field], structure)
        
        #We can optimize the find_cluster function for the refactor! No need to return more than cluster_ids , no need to bring it into awkward again
        inter = self.find_cluster(inter, self.micro_separation/10, self.micro_separation_time)
        cluster_ids = inter['cluster_ids']
        
        len_output = len(epix.awkward_to_flat_numpy(cluster_ids))
        numpy_data = np.zeros(len_output, dtype=self.dtype)
        numpy_data["cluster_ids"] = epix.awkward_to_flat_numpy(cluster_ids)
        
        numpy_data["time"] = geant4_interactions["time"]
        numpy_data["endtime"] = geant4_interactions["endtime"]
        
        return numpy_data
    
    def find_cluster(self, interactions, cluster_size_space, cluster_size_time):
        """
        Function which finds cluster within a event.
        Args:
            x (pandas.DataFrame): Subentries of event must contain the
                fields, x,y,z,time
            cluster_size_space (float): Max spatial distance between two points to
                be inside a cluster [cm].
            cluster_size_time (float): Max time distance between two points to be 
                inside a cluster [ns].

        Returns:
            awkward.array: Adds to interaction a cluster_ids record.
        """
        # TODO is there a better way to get the df?
        df = []
        for key in ['x', 'y', 'z', 'ed', 't']:
            df.append(ak.to_pandas(interactions[key], anonymous=key))
        df = pd.concat(df, axis=1)

        if df.empty:
            # TPC interaction is empty
            return interactions

        # Splitting into individual events and apply time clustering:
        groups = df.groupby('entry')

        df["time_cluster"] = np.concatenate(groups.apply(lambda x: simple_1d_clustering(x.t.values, cluster_size_time)))

        # Splitting into individual events and time cluster and apply space clustering space:
        df['cluster_id'] = np.zeros(len(df.index), dtype=np.int)

        for evt in df.index.get_level_values(0).unique():
            _df_evt = df.loc[evt]
            _t_clusters = _df_evt.time_cluster.unique()
            add_to_cluster = 0

            for _t in _t_clusters:
                _cl = self._find_cluster(_df_evt[_df_evt.time_cluster == _t], cluster_size_space=cluster_size_space)
                df.loc[(df.time_cluster == _t) & (df.index.get_level_values(0) == evt), 'cluster_id'] = _cl + add_to_cluster
                add_to_cluster = max(_cl) + add_to_cluster + 1

        ci = df.loc[:, 'cluster_id'].values
        offsets = ak.num(interactions['x'])
        interactions['cluster_ids'] = reshape_awkward(ci, offsets)

        return interactions


    def _find_cluster(self, x, cluster_size_space):
        """
        Function which finds cluster within a event.
        Args:
            x (pandas.DataFrame): Subentries of event must contain the
                fields, x,y,z,time
            cluster_size_space (float): Max distance between two points to
                be inside a cluster [cm].
        Returns:
            functon: to be used in groupby.apply.
        """
        db_cluster = DBSCAN(eps=cluster_size_space, min_samples=1)
        xprime = x[['x', 'y', 'z']].values
        return db_cluster.fit_predict(xprime)
    
@numba.jit(nopython=True)
def simple_1d_clustering(data, scale):
    """
    Function to cluster one dimensional data.
    Args:
        data (numpy.array): one dimensional array to be clusterd
        scale (float): Max distance between two points to
            be inside a cluster.

    Returns:
        clusters_undo_sort (np.array): Cluster Labels
    """

    idx_sort = np.argsort(data)
    idx_undo_sort = np.argsort(idx_sort)

    data_sorted = data[idx_sort]

    diff = data_sorted[1:] - data_sorted[:-1]

    clusters = [0]
    c = 0
    for value in diff:
        if value <= scale:
            clusters.append(c)
        elif value > scale:
            c = c + 1
            clusters.append(c)

    clusters_undo_sort = np.array(clusters)[idx_undo_sort]

    return clusters_undo_sort
