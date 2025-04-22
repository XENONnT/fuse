import os
import shutil
import unittest
import tempfile
import timeout_decorator
import fuse
import utilix
from _utils import test_root_file_name

TIMEOUT = 60


class TestMicroPhysicsBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()

        cls.test_context = fuse.context.microphysics_context(cls.temp_dir.name)

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
        downloader = utilix.mongo_storage.MongoDownloader(store_files_at=(self.temp_dir.name,))
        downloader.download_single(test_root_file_name, human_readable_file_name=True)

        assert os.path.exists(os.path.join(self.temp_dir.name, test_root_file_name))

    def tearDown(self):
        # self.temp_dir.cleanup()
        shutil.rmtree(self.temp_dir.name)
        os.makedirs(self.temp_dir.name)


class TestMicroPhysics(TestMicroPhysicsBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @timeout_decorator.timeout(TIMEOUT, exception_message="ChunkInput timed out")
    def test_ChunkInput(self):
        self.test_context.make(self.run_number, "geant4_interactions")

    @timeout_decorator.timeout(TIMEOUT, exception_message="FindCluster timed out")
    def test_FindCluster(self):
        self.test_context.make(self.run_number, "cluster_index")

    @timeout_decorator.timeout(TIMEOUT, exception_message="MergeCluster timed out")
    def test_MergeCluster(self):
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


class TestMicroPhysicsAlternativePlugins(TestMicroPhysicsBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @timeout_decorator.timeout(TIMEOUT, exception_message="BBFYields timed out")
    def test_BBFYields(self):
        self.test_context.register(fuse.plugins.BBFYields)
        self.test_context.make(self.run_number, "quanta")

    @timeout_decorator.timeout(TIMEOUT, exception_message="GasPhasePlugin timed out")
    def test_GasPhasePlugin(self):
        self.test_context.register(fuse.plugins.XENONnT_GasPhase)
        self.test_context.make(self.run_number, "gas_phase_interactions")


if __name__ == "__main__":
    unittest.main()
