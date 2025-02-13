import os
import shutil
import unittest
import tempfile
import timeout_decorator
import fuse
from _utils import build_random_instructions

TIMEOUT = 480


class TestDetectorPhysicsCsv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()

        cls.test_context = fuse.context.full_chain_context(
            output_folder=cls.temp_dir.name, run_without_proper_corrections=True
        )
        cls.test_context.register(fuse.plugins.detector_physics.ChunkCsvInput)
        cls.test_context.deregister_plugins_with_missing_dependencies()

        cls.input_file = os.path.join(cls.temp_dir.name, "test.csv")

        cls.test_context.set_config(
            {
                "input_file": cls.input_file,
                "entry_stop": 5,
            }
        )

        cls.run_number = "TestRun_00000"

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def setUp(self):
        detectorphysics_instructions = build_random_instructions(10)
        detectorphysics_instructions.to_csv(self.input_file, index=False)

        assert os.path.exists(self.input_file)

    def tearDown(self):
        # self.temp_dir.cleanup()
        shutil.rmtree(self.temp_dir.name)
        os.makedirs(self.temp_dir.name)

    @timeout_decorator.timeout(TIMEOUT, exception_message="ChunkCsvInput timed out")
    def test_ChunkCsvInput(self):
        self.test_context.make(self.run_number, "microphysics_summary")

    @timeout_decorator.timeout(
        TIMEOUT, exception_message="PMTResponseAndDAQ with CSV input timed out"
    )
    def test_PMTResponseAndDAQ_CsvInput(self):
        self.test_context.make(self.run_number, "raw_records")


if __name__ == "__main__":
    unittest.main()
