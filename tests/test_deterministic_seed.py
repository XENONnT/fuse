import unittest
import tempfile
import numpy as np
import fuse
import straxen
from numpy.testing import assert_array_equal, assert_raises
from _utils import test_root_file_name


class TestDeterministicSeed(unittest.TestCase):

    def setUp(self):

        self.temp_dir_0 = tempfile.TemporaryDirectory()
        self.temp_dir_1 = tempfile.TemporaryDirectory()

        downloader = straxen.MongoDownloader(store_files_at=(self.temp_dir_0.name,))
        downloader.download_single(test_root_file_name, human_readable_file_name=True)

        self.test_context_0 = fuse.context.full_chain_context(output_folder = self.temp_dir_0.name)

        self.test_context_0.set_config({"path": self.temp_dir_0.name,
                                      "file_name": test_root_file_name,
                                      "entry_stop": 5,
                                      })

        self.test_context_1 = fuse.context.full_chain_context(output_folder = self.temp_dir_1.name)

        self.test_context_1.set_config({"path": self.temp_dir_1.name,
                                      "file_name": test_root_file_name,
                                      "entry_stop": 5,
                                      })
        
        self.run_number_0 = "TestRun_00000"
        self.run_number_1 = "TestRun_00001"

    def tearDown(self):

        self.temp_dir_0.cleanup()
        self.temp_dir_1.cleanup()

    def test_MicroPhysics_SameSeed(self):
        """Test that the same run_number and lineage produce the same random seed and thus the same output"""

        self.test_context_0.make(self.run_number_0, "microphysics_summary")
        self.test_context_1.make(self.run_number_1, "microphysics_summary")

        output_0 = self.test_context_0.get_array(self.run_number_0, "microphysics_summary", progress_bar=False)
        output_1 = self.test_context_1.get_array(self.run_number_0, "microphysics_summary", progress_bar=False)

        assert_array_equal(output_0, output_1)

    def test_MicroPhysics_DifferentSeed(self):
        """Test that a different run_number produce a different random seed and thus different output"""

        self.test_context_0.make(self.run_number_0, "microphysics_summary")
        self.test_context_1.make(self.run_number_1, "microphysics_summary")

        output_0 = self.test_context_0.get_array(self.run_number_0, "microphysics_summary", progress_bar=False)
        output_1 = self.test_context_1.get_array(self.run_number_1, "microphysics_summary", progress_bar=False)

        assert_raises(AssertionError, assert_array_equal, output_0, output_1)

    def test_FullChain_SameSeed(self):
        """Test that the same run_number and lineage produce the same random seed and thus the same output"""

        self.test_context_0.make(self.run_number_0, "raw_records")
        self.test_context_1.make(self.run_number_1, "raw_records")

        output_0 = self.test_context_0.get_array(self.run_number_0, "raw_records", progress_bar=False)
        output_1 = self.test_context_1.get_array(self.run_number_0, "raw_records", progress_bar=False)

        assert_array_equal(output_0, output_1)

    def test_FullChain_DifferentSeed(self):
        """Test that a different run_number produce a different random seed and thus different output"""

        self.test_context_0.make(self.run_number_0, "raw_records")
        self.test_context_1.make(self.run_number_1, "raw_records")

        output_0 = self.test_context_0.get_array(self.run_number_0, "raw_records", progress_bar=False)
        output_1 = self.test_context_1.get_array(self.run_number_1, "raw_records", progress_bar=False)

        assert_raises(AssertionError, assert_array_equal, output_0, output_1)

if __name__ == '__main__':
    unittest.main()