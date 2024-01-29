import unittest
import fuse
import tempfile
import timeout_decorator

TIMEOUT = 60

class TestMicroPhysics(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.temp_dir = tempfile.TemporaryDirectory()

        self.test_context = fuse.context.microphysics_context(self.temp_dir.name)

        self.test_context.set_config({"path": "/project2/lgrandi/xenonnt/simulations/testing",
                                      "file_name": "pmt_neutrons_100.root",
                                      "entry_stop": 25,
                                      })
        
        self.run_number = "TestRun_00000"

    @classmethod
    def tearDownClass(self):

        self.temp_dir.cleanup()
    
    @timeout_decorator.timeout(TIMEOUT, exception_message='ChunkInput timed out')
    def test_ChunkInput(self):

        self.test_context.make(self.run_number, "geant4_interactions")

    @timeout_decorator.timeout(TIMEOUT, exception_message='FindCluster timed out')
    def test_FindCluster(self):

        self.test_context.make(self.run_number, "cluster_index")

    @timeout_decorator.timeout(TIMEOUT, exception_message='MergeCluster timed out')
    def test_MergeCluster(self):

        self.test_context.make(self.run_number, "clustered_interactions")

    @timeout_decorator.timeout(TIMEOUT, exception_message='VolumesMerger timed out')
    def test_VolumesMerger(self):

        self.test_context.make(self.run_number, "interactions_in_roi")

    @timeout_decorator.timeout(TIMEOUT, exception_message='ElectricField timed out')
    def test_ElectricField(self):

        self.test_context.make(self.run_number, "electric_field_values")

    @timeout_decorator.timeout(TIMEOUT, exception_message='NestYields timed out')
    def test_NestYields(self):

        self.test_context.make(self.run_number, "quanta")

    @timeout_decorator.timeout(TIMEOUT, exception_message='MicroPhysicsSummary timed out')
    def test_MicroPhysicsSummary(self):

        self.test_context.make(self.run_number, "microphysics_summary")
        
if __name__ == '__main__':
    unittest.main()