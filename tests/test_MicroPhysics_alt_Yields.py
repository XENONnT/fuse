import os
import shutil
import unittest
import tempfile
import timeout_decorator
import fuse
import straxen
from _utils import test_root_file_name
import pickle

TIMEOUT = 60


def yields_dummy_func(x):
    """Dummy function that returns two values for the n_photon and n_electron.

    To be used as a dummy function for the BetaYields plugin. Needs to
    be defined outside the test class to be picklable.
    """
    return 40


class TestAlternativeYields(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()

        cls.test_context = fuse.context.full_chain_context(
            output_folder=cls.temp_dir.name, run_without_proper_corrections=True
        )

        cls.test_context.set_config(
            {
                "path": cls.temp_dir.name,
                "file_name": test_root_file_name,
                "entry_stop": 25,
            }
        )

        cls.run_number = "TestRun_00000"

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def setUp(self):
        downloader = straxen.MongoDownloader(store_files_at=(self.temp_dir.name,))
        downloader.download_single(test_root_file_name, human_readable_file_name=True)

        assert os.path.exists(os.path.join(self.temp_dir.name, test_root_file_name))

    def tearDown(self):
        shutil.rmtree(self.temp_dir.name)
        os.makedirs(self.temp_dir.name)

    @timeout_decorator.timeout(TIMEOUT, exception_message="BetaYields timed out")
    def test_BetaYields(self):
        self.test_context.register(fuse.BetaYields)

        # Make a dummy pkl file, a function that returns two values
        # one for the photon yield and one for the electron yield
        # as a function of energy
        spline_func_name = os.path.join(self.temp_dir.name, "beta_quanta_spline.pkl")
        with open(spline_func_name, "wb") as f:
            pickle.dump((yields_dummy_func, yields_dummy_func), f)

        self.test_context.set_config({"beta_quanta_spline": spline_func_name})

        # Make the plugin
        self.test_context.make(self.run_number, "quanta")

    # Add tests for the BBF yields later


if __name__ == "__main__":
    unittest.main()
