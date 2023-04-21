import numpy as np
import pandas as pd
import numba
import strax
import awkward as ak
import logging

from ...common import reshape_awkward, awkward_to_flat_numpy

from sklearn.cluster import DBSCAN

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('XeSim.micro_physics.find_cluster')
log.setLevel('WARNING')

@strax.takes_config(
    strax.Option('micro_separation', default=0.005, track=False, infer_type=False,
                 help="DBSCAN clustering distance (mm)"),
    strax.Option('micro_separation_time', default = 10, track=False, infer_type=False,
                 help="Clustering time (ns)"),
    strax.Option('debug', default=False, track=False, infer_type=False,
                 help="Show debug informations"),
)
class FindCluster(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ("geant4_interactions",)
    
    provides = "cluster_index"
    
    dtype = [('cluster_ids', np.int64),
            ]
    dtype = dtype + strax.time_fields

    #Forbid rechunking
    rechunk_on_save = False

    def setup(self):
        
        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running FindCluster in debug mode")
    
    def compute(self, geant4_interactions):
        """
        Compute the cluster IDs for a set of GEANT4 interactions.

        Args:
            geant4_interactions (np.ndarray): An array of GEANT4 interaction data.

        Returns:
            np.ndarray: An array of cluster IDs with corresponding time and endtime values.
        """
        if len(geant4_interactions) == 0:
            return np.zeros(0, dtype=self.dtype)

        inter = ak.from_numpy(np.empty(1, dtype=geant4_interactions.dtype))
        structure = np.unique(geant4_interactions['evtid'], return_counts=True)[1]

        for field in inter.fields:
            inter[field] = reshape_awkward(geant4_interactions[field], structure)

        # We can optimize the find_cluster function for the refactor!
        # No need to return more than cluster_ids,
        # no need to bring it into awkward again
        inter = self.find_cluster(inter, self.micro_separation / 10,
                                  self.micro_separation_time)
        cluster_ids = inter['cluster_ids']

        len_output = len(awkward_to_flat_numpy(cluster_ids))
        numpy_data = np.zeros(len_output, dtype=self.dtype)
        numpy_data["cluster_ids"] = awkward_to_flat_numpy(cluster_ids)

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
        for key in ['x', 'y', 'z', 'ed', 'time']:
            df.append(ak.to_pandas(interactions[key], anonymous=key))
        df = pd.concat(df, axis=1)

        if df.empty:
            # TPC interaction is empty
            return interactions

        # Splitting into individual events and apply time clustering:
        groups = df.groupby('entry')

        df["time_cluster"] = np.concatenate(groups.apply(lambda x: simple_1d_clustering(x.time.values, cluster_size_time)))

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

    @staticmethod
    def _find_cluster(x, cluster_size_space):
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