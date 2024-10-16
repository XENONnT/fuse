import os
import shutil
import unittest
import tempfile
import timeout_decorator
import fuse
import utilix
from _utils import test_root_file_name

TIMEOUT = 240


class TestLineageClustering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()

        cls.test_context = fuse.context.full_chain_context(
            output_folder=cls.temp_dir.name,
            run_without_proper_corrections=True,
            clustering_method="lineage",
        )

        cls.test_context.set_config(
            {
                "path": cls.temp_dir.name,
                "file_name": test_root_file_name,
                # "entry_stop": 5,
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

    def tearDown(self):
        # self.temp_dir.cleanup()
        shutil.rmtree(self.temp_dir.name)
        os.makedirs(self.temp_dir.name)

    @timeout_decorator.timeout(TIMEOUT, exception_message="ChunkInput timed out")
    def test_ChunkInput(self):
        self.test_context.make(self.run_number, "geant4_interactions")

    @timeout_decorator.timeout(TIMEOUT, exception_message="LineageClustering timed out")
    def test_LineageClustering(self):
        self.test_context.make(self.run_number, "interaction_lineage")

    @timeout_decorator.timeout(TIMEOUT, exception_message="MergeLineage timed out")
    def test_MergeLineage(self):
        self.test_context.make(self.run_number, "clustered_interactions")

    @timeout_decorator.timeout(TIMEOUT, exception_message="VolumesMerger timed out")
    def test_VolumesMerger(self):
        self.test_context.make(self.run_number, "interactions_in_roi")

    @timeout_decorator.timeout(TIMEOUT, exception_message="ElectricField timed out")
    def test_ElectricField(self):
        self.test_context.make(self.run_number, "electric_field_values")

    @timeout_decorator.timeout(TIMEOUT, exception_message="NestYields timed out")
    def test_NestYields(self):
        self.test_context.make(self.run_number, "quanta")

    @timeout_decorator.timeout(TIMEOUT, exception_message="MicroPhysicsSummary timed out")
    def test_MicroPhysicsSummary(self):
        self.test_context.make(self.run_number, "microphysics_summary")


if __name__ == "__main__":
    unittest.main()
