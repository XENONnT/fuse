import os
import shutil
import unittest
import tempfile
import timeout_decorator
import fuse
import straxen
from _utils import test_root_file_name

TIMEOUT = 240


class TestFullChain(unittest.TestCase):
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
        downloader = straxen.MongoDownloader(store_files_at=(self.temp_dir.name,))
        downloader.download_single(test_root_file_name, human_readable_file_name=True)

        assert os.path.exists(os.path.join(self.temp_dir.name, test_root_file_name))

    def tearDown(self):
        # self.temp_dir.cleanup()
        shutil.rmtree(self.temp_dir.name)
        os.makedirs(self.temp_dir.name)

    @timeout_decorator.timeout(TIMEOUT, exception_message="S1PhotonHits timed out")
    def test_S1PhotonHits(self):
        self.test_context.make(self.run_number, "s1_photons")

    @timeout_decorator.timeout(TIMEOUT, exception_message="S1PhotonPropagation timed out")
    def test_S1PhotonPropagation(self):
        self.test_context.make(self.run_number, "propagated_s1_photons")

    @timeout_decorator.timeout(TIMEOUT, exception_message="ElectronDrift timed out")
    def test_ElectronDrift(self):
        self.test_context.make(self.run_number, "drifted_electrons")

    @timeout_decorator.timeout(TIMEOUT, exception_message="ElectronExtraction timed out")
    def test_ElectronExtraction(self):
        self.test_context.make(self.run_number, "extracted_electrons")

    @timeout_decorator.timeout(TIMEOUT, exception_message="ElectronTiming timed out")
    def test_ElectronTiming(self):
        self.test_context.make(self.run_number, "electron_time")

    @timeout_decorator.timeout(TIMEOUT, exception_message="SecondaryScintillation timed out")
    def test_SecondaryScintillation(self):
        self.test_context.make(self.run_number, "s2_photons")
        # self.test_context.make(self.run_number, "s2_photons_sum")

    @timeout_decorator.timeout(TIMEOUT, exception_message="S2PhotonPropagation timed out")
    def test_S2PhotonPropagation(self):
        self.test_context.make(self.run_number, "propagated_s2_photons")

    @timeout_decorator.timeout(TIMEOUT, exception_message="PMTAfterPulses timed out")
    def test_PMTAfterPulses(self):
        self.test_context.make(self.run_number, "pmt_afterpulses")

    @timeout_decorator.timeout(TIMEOUT, exception_message="PulseWindow timed out")
    def test_PulseWindow(self):
        self.test_context.make(self.run_number, "pulse_windows")
        # self.test_context.make(self.run_number, "pulse_ids")

    @timeout_decorator.timeout(TIMEOUT, exception_message="PMTResponseAndDAQ timed out")
    def test_PMTResponseAndDAQ(self):
        self.test_context.make(self.run_number, "raw_records")


if __name__ == "__main__":
    unittest.main()
