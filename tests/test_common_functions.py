import numpy as np
import awkward as ak
import unittest
from fuse.common import awkward_to_flat_numpy, full_array_to_numpy, dynamic_chunking


class TestFullArrayToNumpy(unittest.TestCase):
    def test_full_array_to_numpy(self):
        array = ak.Array(
            {
                "x": [[1, 2, 3], [4, 5], [6]],
                "y": [[7, 8, 9], [10, 11], [12]],
                "z": [[13, 14, 15], [16, 17], [18]],
            }
        )
        dtype = np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])
        expected_output = np.zeros(6, dtype=dtype)
        expected_output["x"] = [1, 2, 3, 4, 5, 6]
        expected_output["y"] = [7, 8, 9, 10, 11, 12]
        expected_output["z"] = [13, 14, 15, 16, 17, 18]

        result = full_array_to_numpy(array, dtype)

        np.testing.assert_array_equal(result, expected_output)


class TestAwkwardToFlatNumpy(unittest.TestCase):
    def test_empty_array(self):
        # Test when the input array is empty
        array = ak.Array([])
        result = awkward_to_flat_numpy(array)
        expected = np.array([])
        np.testing.assert_allclose(result, expected)

    def test_single_jagged_layer(self):
        # Test when the input array has a single jagged layer
        array = ak.Array([[1, 2, 3], [4, 5], [6]])
        result = awkward_to_flat_numpy(array)
        expected = np.array([1, 2, 3, 4, 5, 6])
        np.testing.assert_allclose(result, expected)


class TestDynamicChunking(unittest.TestCase):
    def test_two_chunks(self):
        data = np.array([1, 2, 3, 4, 7, 8, 9, 10])
        scale = 2
        n_min = 2
        expected_clusters = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        clusters = dynamic_chunking(data, scale, n_min)
        np.testing.assert_array_equal(clusters, expected_clusters)

    def test_single_chunks(self):
        data = np.array([1, 2, 3, 4])
        scale = 2
        n_min = 2
        expected_clusters = np.array([0, 0, 0, 0])
        clusters = dynamic_chunking(data, scale, n_min)
        np.testing.assert_array_equal(clusters, expected_clusters)

    def test_chunk_extension(self):
        data = np.array([1, 2, 3, 4, 7, 8, 9, 10])
        scale = 2
        n_min = 5
        expected_clusters = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        clusters = dynamic_chunking(data, scale, n_min)
        np.testing.assert_array_equal(clusters, expected_clusters)


if __name__ == "__main__":
    unittest.main()
