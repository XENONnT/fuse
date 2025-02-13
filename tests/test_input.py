import os
import shutil
import unittest
import tempfile
import timeout_decorator
import fuse
import utilix
import numpy as np
from _utils import test_root_file_name

TIMEOUT = 60


class TestInput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.run_number = "TestRun_00000"

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def setUp(self):
        downloader = utilix.mongo_storage.MongoDownloader(store_files_at=(self.temp_dir.name,))
        downloader.download_single(test_root_file_name, human_readable_file_name=True)

        assert os.path.exists(os.path.join(self.temp_dir.name, test_root_file_name))

    def tearDown(self):
        # self.temp_dir.cleanup()
        shutil.rmtree(self.temp_dir.name)
        os.makedirs(self.temp_dir.name)

    @timeout_decorator.timeout(TIMEOUT, exception_message="LoadAll timed out")
    def test_load_all(self):
        test_context = fuse.context.microphysics_context(self.temp_dir.name)
        test_context.set_config(
            {
                "path": self.temp_dir.name,
                "file_name": test_root_file_name,
            }
        )
        g4_loaded = test_context.get_array(self.run_number, "geant4_interactions")
        loaded_event_count = len(np.unique(g4_loaded["eventid"]))
        self.assertTrue(
            loaded_event_count == 52, f"Expecting 52 events, but got {loaded_event_count} events"
        )

    @timeout_decorator.timeout(TIMEOUT, exception_message="LoadHalf timed out")
    def test_load_half(self):
        test_context = fuse.context.microphysics_context(self.temp_dir.name)
        test_context.set_config(
            {
                "path": self.temp_dir.name,
                "file_name": test_root_file_name,
                "entry_start": 0,
                "entry_stop": 50,
            }
        )
        g4_loaded = test_context.get_array(self.run_number, "geant4_interactions")
        loaded_event_count = len(np.unique(g4_loaded["eventid"]))

        self.assertTrue(
            loaded_event_count == 26, f"Expecting 26 events, but got {loaded_event_count} events"
        )

    @timeout_decorator.timeout(TIMEOUT, exception_message="LoadEventIDAll timed out")
    def test_load_eventid_all(self):
        test_context = fuse.context.microphysics_context(self.temp_dir.name)
        test_context.set_config(
            {"path": self.temp_dir.name, "file_name": test_root_file_name, "cut_by_eventid": True}
        )
        g4_loaded = test_context.get_array(self.run_number, "geant4_interactions")
        loaded_event_count = len(np.unique(g4_loaded["eventid"]))
        self.assertTrue(
            loaded_event_count == 52, f"Expecting 52 events, but got {loaded_event_count} events"
        )

    @timeout_decorator.timeout(TIMEOUT, exception_message="LoadEventIDHalf timed out")
    def test_load_eventid_half(self):
        test_context = fuse.context.microphysics_context(self.temp_dir.name)
        test_context.set_config(
            {
                "path": self.temp_dir.name,
                "file_name": test_root_file_name,
                "cut_by_eventid": True,
                "entry_start": 0,
                "entry_stop": 50,
            }
        )
        g4_loaded = test_context.get_array(self.run_number, "geant4_interactions")
        loaded_event_count = len(np.unique(g4_loaded["eventid"]))
        self.assertTrue(
            loaded_event_count == 23, f"Expecting 23 events, but got {loaded_event_count} events"
        )

    @timeout_decorator.timeout(TIMEOUT, exception_message="InvalidArgs0 timed out")
    def test_invalid_args_0(self):
        test_context = fuse.context.microphysics_context(self.temp_dir.name)
        test_context.set_config(
            {
                "path": self.temp_dir.name,
                "file_name": test_root_file_name,
                "cut_by_eventid": False,
                "entry_start": 75,
                "entry_stop": 15,
            }
        )
        with self.assertRaises(ValueError):
            test_context.make(self.run_number, "geant4_interactions")

    @timeout_decorator.timeout(TIMEOUT, exception_message="InvalidArgs1 timed out")
    def test_invalid_args_1(self):
        test_context = fuse.context.microphysics_context(self.temp_dir.name)
        test_context.set_config(
            {
                "path": self.temp_dir.name,
                "file_name": test_root_file_name,
                "cut_by_eventid": False,
                "entry_start": 90,
                "entry_stop": 91,
            }
        )
        with self.assertRaises(ValueError):
            test_context.make(self.run_number, "geant4_interactions")

    @timeout_decorator.timeout(TIMEOUT, exception_message="InvalidArgs2 timed out")
    def test_invalid_args_2(self):
        test_context = fuse.context.microphysics_context(self.temp_dir.name)
        test_context.set_config(
            {
                "path": self.temp_dir.name,
                "file_name": test_root_file_name,
                "cut_by_eventid": False,
                "entry_start": -2,
                "entry_stop": -1,
            }
        )
        with self.assertRaises(ValueError):
            test_context.make(self.run_number, "geant4_interactions")

    @timeout_decorator.timeout(TIMEOUT, exception_message="InvalidArgs3 timed out")
    def test_invalid_args_3(self):
        test_context = fuse.context.microphysics_context(self.temp_dir.name)
        test_context.set_config(
            {
                "path": self.temp_dir.name,
                "file_name": test_root_file_name,
                "cut_by_eventid": True,
                "entry_start": -2,
                "entry_stop": -1,
            }
        )
        with self.assertRaises(ValueError):
            test_context.make(self.run_number, "geant4_interactions")

    @timeout_decorator.timeout(TIMEOUT, exception_message="InvalidArgs4 timed out")
    def test_invalid_args_4(self):
        test_context = fuse.context.microphysics_context(self.temp_dir.name)
        test_context.set_config(
            {
                "path": self.temp_dir.name,
                "file_name": test_root_file_name,
                "cut_by_eventid": True,
                "entry_start": 102,
                "entry_stop": 103,
            }
        )
        with self.assertRaises(ValueError):
            test_context.make(self.run_number, "geant4_interactions")

    @timeout_decorator.timeout(TIMEOUT, exception_message="InvalidArgs5 timed out")
    def test_invalid_args_5(self):
        test_context = fuse.context.microphysics_context(self.temp_dir.name)
        test_context.set_config(
            {
                "path": self.temp_dir.name,
                "file_name": test_root_file_name,
                "cut_by_eventid": True,
                "entry_start": 10,
                "entry_stop": 11,
            }
        )
        with self.assertRaises(ValueError):
            test_context.make(self.run_number, "geant4_interactions")


if __name__ == "__main__":
    unittest.main()
