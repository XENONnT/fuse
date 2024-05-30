import unittest
import numpy as np
from fuse.plugins.micro_physics.find_cluster import _find_cluster, simple_1d_clustering


class TestFindCluster(unittest.TestCase):

    def test__find_cluster_all_separate(self):

        dtype = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)])
        x = np.zeros(6, dtype=dtype)
        x["x"] = [1, 2, 3, 4, 5, 6]
        x["y"] = [1, 2, 3, 4, 5, 6]
        x["z"] = [1, 2, 3, 4, 5, 6]

        cluster_size_space = 0.1
        expected_result = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
        result = _find_cluster(x, cluster_size_space)
        np.testing.assert_array_equal(result, expected_result)

    def test__find_cluster_all_merged(self):

        dtype = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)])
        x = np.zeros(6, dtype=dtype)
        x["x"] = [1, 2, 3, 4, 5, 6]
        x["y"] = [1, 2, 3, 4, 5, 6]
        x["z"] = [1, 2, 3, 4, 5, 6]

        cluster_size_space = 1.8
        expected_result = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
        result = _find_cluster(x, cluster_size_space)
        np.testing.assert_array_equal(result, expected_result)

    def test__find_cluster_two_clusters(self):

        dtype = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)])
        x = np.zeros(7, dtype=dtype)
        x["x"] = [1, 2, 3, 7, 5, 6, 2.5]
        x["y"] = [1, 2, 3, 7, 5, 6, 2.5]
        x["z"] = [1, 2, 3, 7, 5, 6, 2.5]

        cluster_size_space = 1.8
        expected_result = np.array([0, 0, 0, 1, 1, 1, 0], dtype=np.int32)
        result = _find_cluster(x, cluster_size_space)
        np.testing.assert_array_equal(result, expected_result)

    def test_simple_1d_clustering_two_clusters_ordered_input(self):
        data = np.array([0, 1, 2, 4, 5], dtype=np.float32)
        scale = 1
        expected_result = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        result = simple_1d_clustering(data, scale)
        np.testing.assert_array_equal(result, expected_result)

    def test_simple_1d_clustering_three_clusters_unordered_input(self):
        data = np.array([4, 1, 2, 10, 5], dtype=np.float32)
        scale = 1
        expected_result = np.array([1, 0, 0, 2, 1], dtype=np.int32)
        result = simple_1d_clustering(data, scale)
        np.testing.assert_array_equal(result, expected_result)


if __name__ == "__main__":
    unittest.main()
