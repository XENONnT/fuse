import unittest
import fuse
import tempfile
import os
import numpy as np

class TestFindCluster(unittest.TestCase):

    def setUp(self):

        self.temp_dir = tempfile.TemporaryDirectory()

        self.test_context = fuse.context.microphysics_context(self.temp_dir.name)
        
        self.plugin = self.test_context.get_single_plugin("TestRun_00000", "cluster_index")

    def tearDown(self):
            
        self.temp_dir.cleanup()

    def test_ChunkInput(self):
            
        plugin_input = np.load(os.path.join(os.getcwd(), "data", "geant4_interactions_for_test.npy"))

        plugin_output = self.plugin.compute(plugin_input)

        true_output = np.load(os.path.join(os.getcwd(), "data", "cluster_index_for_test.npy"))

        self.assertTrue(np.all(plugin_output == true_output))

if __name__ == '__main__':
    unittest.main()