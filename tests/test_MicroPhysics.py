import os
import shutil
import unittest
import tempfile
import fuse
import straxen
from _utils import test_root_file_name


class TestMicroPhysics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.temp_dir = tempfile.TemporaryDirectory()

        downloader = straxen.MongoDownloader(store_files_at=(cls.temp_dir.name,))
        downloader.download_single(test_root_file_name, human_readable_file_name=True)

        assert os.path.exists(os.path.join(cls.temp_dir.name, test_root_file_name))

        cls.test_context = fuse.context.microphysics_context(cls.temp_dir.name)

        cls.test_context.set_config({"path": cls.temp_dir.name,
                                      "file_name": test_root_file_name,
                                      "entry_stop": 25,
                                      })
        
        cls.run_number = "TestRun_00000"

    @classmethod
    def tearDownClass(cls):

        cls.temp_dir.cleanup()

    def tearDown(self):

        # self.temp_dir.cleanup()
        shutil.rmtree(self.temp_dir.name)
        os.makedirs(self.temp_dir.name)
    
    def test_ChunkInput(self):

        self.test_context.make(self.run_number, "geant4_interactions")

    def test_FindCluster(self):

        self.test_context.make(self.run_number, "cluster_index")

    def test_MergeCluster(self):

        self.test_context.make(self.run_number, "clustered_interactions")

    def test_VolumesMerger(self):

        self.test_context.make(self.run_number, "interactions_in_roi")

    def test_ElectricField(self):

        self.test_context.make(self.run_number, "electric_field_values")

    def test_NestYields(self):

        self.test_context.make(self.run_number, "quanta")

    def test_MicroPhysicsSummary(self):

        self.test_context.make(self.run_number, "microphysics_summary")
        
if __name__ == '__main__':
    unittest.main()