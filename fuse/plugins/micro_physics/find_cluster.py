import numpy as np
import pandas as pd
import numba
import strax
import awkward as ak
import logging
import straxen
from sklearn.cluster import DBSCAN

export, __all__ = strax.exporter()

from ...common import FUSE_PLUGIN_TIMEOUT

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.micro_physics.find_cluster')

@export
class FindCluster(strax.Plugin):
    
    __version__ = "0.1.0"
    
    depends_on = ("geant4_interactions")
    
    provides = "cluster_index"
    
    dtype = [('cluster_ids', np.int64),
            ]
    dtype = dtype + strax.time_fields

    #Forbid rechunking
    rechunk_on_save = False

    save_when = strax.SaveWhen.TARGET

    input_timeout = FUSE_PLUGIN_TIMEOUT

    #Config options
    debug = straxen.URLConfig(
        default=False, type=bool,track=False,
        help='Show debug informations',
    )

    micro_separation_time = straxen.URLConfig(
        default=10, type=(int, float),
        help='Clustering time (ns)',
    )

    micro_separation = straxen.URLConfig(
        default=0.005, type=(int, float),
        help='DBSCAN clustering distance (mm)',
    )

    def setup(self):
        
        if self.debug:
            log.setLevel('DEBUG')
            log.debug(f"Running FindCluster version {self.__version__} in debug mode")
        else: 
            log.setLevel('WARNING')
    
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

        cluster_ids = self.find_cluster(geant4_interactions, self.micro_separation / 10,
                                  self.micro_separation_time)

        numpy_data = np.zeros(len(geant4_interactions), dtype=self.dtype)
        numpy_data["cluster_ids"] = cluster_ids

        numpy_data["time"] = geant4_interactions["time"]
        numpy_data["endtime"] = geant4_interactions["endtime"]

        return numpy_data
    
    @staticmethod
    def find_cluster(interactions, cluster_size_space, cluster_size_time):
        """
        Function to find clusters in a set of interactions. First interactions are clustered in time, then in space.
        """
        
        time_cluster = simple_1d_clustering(interactions["time"], cluster_size_time)

        # Splitting into time cluster and apply space clustering space:
        cluster_id = np.zeros(len(interactions), dtype=np.int32)
        spacial_cluster = np.zeros(len(interactions), dtype=np.int32)

        _t_clusters = np.unique(time_cluster)
        for _t in _t_clusters:
            _cl = _find_cluster(interactions[time_cluster == _t], cluster_size_space=cluster_size_space)
            spacial_cluster[time_cluster == _t] = _cl
        _, cluster_id = np.unique((time_cluster, spacial_cluster), axis=1, return_inverse=True)
        
        return cluster_id

    
def _find_cluster(x, cluster_size_space):
    """
    Function to cluster three dimensional data (x, y, z).
    Args:
        x (np.ndarray): structured numpy array with x, y, z coordinates to be clustered
        cluster_size_space (float): Clustering distance for DBSCAN
    Returns:
        Cluster labels
    """
    db_cluster = DBSCAN(eps=cluster_size_space, min_samples=1)

    #Conversion from numpy structured array to regular array with correct shape for 
    #sklearn somehow works fine via pandas... 
    xprime = pd.DataFrame(x[['x', 'y', 'z']]).values

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