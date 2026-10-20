import unittest

import numpy as np
import strax

from fuse.common import sort_by_time


def _photons(n, rng, n_channels=494, t_span=10_000, min_channel=0):
    x = np.zeros(n, strax.interval_dtype)
    x["time"] = rng.integers(0, t_span, n)
    x["channel"] = rng.integers(min_channel, n_channels, n)
    x["length"] = np.arange(n)  # unique payload, to detect reordering of ties
    x["dt"] = 10
    return x


class TestSortByTime(unittest.TestCase):
    """fuse.common.sort_by_time must give byte-identical output to
    strax.sort_by_time."""

    def assert_same(self, x):
        ref = strax.sort_by_time(x.copy())
        out = sort_by_time(x.copy())
        self.assertEqual(ref.tobytes(), out.tobytes())

    def test_random_with_ties(self):
        rng = np.random.default_rng(1)
        for n in (0, 1, 2, 10, 10_000):
            self.assert_same(_photons(n, rng, t_span=50))

    def test_already_sorted(self):
        rng = np.random.default_rng(2)
        self.assert_same(strax.sort_by_time(_photons(10_000, rng, t_span=100)))

    def test_concatenated_sorted_runs(self):
        rng = np.random.default_rng(3)
        parts = [strax.sort_by_time(_photons(n, rng, t_span=1_000)) for n in (300, 5_000, 50)]
        self.assert_same(np.concatenate(parts))

    def test_negative_channels(self):
        rng = np.random.default_rng(4)
        self.assert_same(_photons(5_000, rng, min_channel=-3, n_channels=10, t_span=20))

    def test_no_channel_field(self):
        rng = np.random.default_rng(5)
        x = np.zeros(5_000, strax.time_fields + [("pulse_id", np.int64)])
        x["time"] = rng.integers(0, 100, len(x))
        x["pulse_id"] = np.arange(len(x))
        self.assert_same(x)
        self.assert_same(strax.sort_by_time(x))

    def test_large_time_range_falls_back(self):
        rng = np.random.default_rng(6)
        x = _photons(1_000, rng)
        x["time"][::2] += np.iinfo(np.int64).max // 100
        self.assert_same(x)


if __name__ == "__main__":
    unittest.main()
