import unittest
import fuse
import tempfile
import os
import numpy as np

class TestChunkInput(unittest.TestCase):

    def setUp(self):

        self.temp_dir = tempfile.TemporaryDirectory()

        self.test_context = fuse.context.microphysics_context(self.temp_dir.name)

        self.test_context.set_config({"path": os.path.join(os.getcwd(), "data"),
                                      "file_name": "microphysics_instructions_for_test.csv",
                                      })
        
        self.run_number = "TestRun_00000"

    def tearDown(self):
            
        self.temp_dir.cleanup()

    def test_ChunkInput(self):
            
        self.test_context.make(self.run_number, "geant4_interactions")
    
        output = self.test_context.get_array(self.run_number, "geant4_interactions")

        true_output = np.load(os.path.join(os.getcwd(), "data", "geant4_interactions_for_test.npy"))

        self.assertTrue(np.all(output == true_output))

if __name__ == '__main__':
    unittest.main()