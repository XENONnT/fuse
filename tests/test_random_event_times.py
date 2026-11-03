import os
import tempfile
import unittest
import logging
import numpy as np
import pandas as pd
from fuse.plugins.micro_physics.input import file_loader


class EarliestTimesRng:
    """Stand-in random generator drawing every event time at the lower edge."""

    def uniform(self, low, high, size):
        return np.full(size, low, dtype=np.float64)


class TestRandomEventTimes(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        n = 3
        df = pd.DataFrame(
            {
                "eventid": np.arange(n),
                "xp": np.zeros(n),
                "yp": np.zeros(n),
                "zp": np.full(n, -500.0),
                "xp_pri": np.zeros(n),
                "yp_pri": np.zeros(n),
                "zp_pri": np.full(n, -500.0),
                "time": np.zeros(n),
                "ed": np.full(n, 10.0),
                "type": ["e-"] * n,
                "trackid": np.ones(n, dtype=int),
                "parenttype": ["none"] * n,
                "parentid": np.zeros(n, dtype=int),
                "creaproc": ["none"] * n,
                "edproc": ["eIoni"] * n,
            }
        )
        self.file_name = "events.csv"
        df.to_csv(os.path.join(self.temp_dir.name, self.file_name), index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_first_chunk_does_not_start_before_zero(self):
        loader = file_loader(
            self.temp_dir.name,
            self.file_name,
            EarliestTimesRng(),
            fixed_event_spacing=False,
            log=logging.getLogger("fuse"),
        )
        chunks = list(loader.output_chunk())
        self.assertGreaterEqual(chunks[0][1], 0)
        for data, left, right, _ in chunks:
            self.assertTrue(np.all((data["time"] >= left) & (data["time"] <= right)))
        self.assertEqual(sum(len(c[0]) for c in chunks), 3)
