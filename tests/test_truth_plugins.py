import os
import shutil
import unittest
import tempfile
import timeout_decorator
import fuse
import utilix
from _utils import test_root_file_name

TIMEOUT = 240


class TestTruthPlugins(unittest.TestCase):
    __test__ = True

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
                "entry_stop": 5,
            }
        )

        cls.run_number = "TestRun_00000"

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def setUp(self):
        downloader = utilix.mongo_storage.MongoDownloader(store_files_at=(self.temp_dir.name,))
        downloader.download_single(test_root_file_name, human_readable_file_name=True)

        assert os.path.exists(os.path.join(self.temp_dir.name, test_root_file_name))

        self.test_context.make(self.run_number, "photon_summary")
        self.test_context.make(self.run_number, "raw_records")

    def tearDown(self):
        # self.temp_dir.cleanup()
        shutil.rmtree(self.temp_dir.name)
        os.makedirs(self.temp_dir.name)

    @timeout_decorator.timeout(TIMEOUT, exception_message="RecordsTruth timed out")
    def test_RecordsTruth(self):
        self.test_context.make(self.run_number, "records_truth")


if __name__ == "__main__":
    unittest.main()
