import numpy as np
import unittest
from fuse.plugins.pmt_and_daq.pmt_response_and_daq import (
    split_photons,
    add_noise,
    add_baseline,
    split_data,
    add_current,
)


class TestWaveformBuildingFunctions(unittest.TestCase):
    def test_split_photons(self):
        propagated_photons = np.array(
            [(1, 10), (1, 20), (2, 30), (2, 40), (2, 50), (3, 60)],
            dtype=[("pulse_id", int), ("time", int)],
        )

        expected_result = [
            np.array([(1, 10), (1, 20)], dtype=[("pulse_id", int), ("time", int)]),
            np.array([(2, 30), (2, 40), (2, 50)], dtype=[("pulse_id", int), ("time", int)]),
            np.array([(3, 60)], dtype=[("pulse_id", int), ("time", int)]),
        ]

        result = split_photons(propagated_photons)

        self.assertEqual(len(result), len(expected_result))
        for i in range(len(result)):
            np.testing.assert_array_equal(result[i], expected_result[i])

    def test_add_noise(self):
        # Test the add_noise function
        array = np.array([1, 2, 3, 4, 5, 6])
        time = 30
        noise_in_channel = np.array([0, 0, 1, 0, 0, 0, 0])

        result = add_noise(array, time, noise_in_channel)

        # Assert the result is correct
        expected_result = np.array([1, 2, 3, 4, 5, 7])
        np.testing.assert_array_equal(result, expected_result)

    def test_add_baseline(self):
        # Test the add_baseline function
        data = np.array([0, 1, 2, 3, 4])
        baseline = 16000

        add_baseline(data, baseline)

        # Assert that the data array has been modified correctly
        np.testing.assert_array_equal(data, np.array([16000, 16001, 16002, 16003, 16004]))

    def test_split_data(self):
        # Test the split_data function
        data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        samples_per_record = 3

        result = split_data(data, samples_per_record)

        # Assert that the result is a tuple of two np.ndarrays
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], int)

        # Assert that the result has the correct shapes
        self.assertEqual(result[0].shape, (4, 3))
        self.assertEqual(result[1], 4)

        np.testing.assert_array_equal(result[0][0], data[0:3])
        np.testing.assert_array_equal(result[0][1], data[3:6])
        np.testing.assert_array_equal(result[0][2], data[6:9])
        np.testing.assert_array_equal(result[0][3], np.array([9, 0, 0]))

    def test_add_current(self):
        photon_timings = np.array([10, 50])
        photon_gains = np.array([100, 200])
        pulse_left = 0
        dt = 10
        pmt_current_templates = [np.array([1, 2, 3, 4, 5])]
        pulse_current = np.zeros(11)

        add_current(
            photon_timings, photon_gains, pulse_left, dt, pmt_current_templates, pulse_current
        )

        # Assert that the pulse_current array has been modified correctly
        expected_result = np.array(
            [0.0, 100.0, 200.0, 300.0, 400.0, 700.0, 400.0, 600.0, 800.0, 1000.0, 0.0]
        )
        np.testing.assert_array_equal(pulse_current, expected_result)


if __name__ == "__main__":
    unittest.main()
