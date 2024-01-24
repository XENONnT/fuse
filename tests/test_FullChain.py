import os
import shutil
import unittest
import fuse
import tempfile

class TestFullChain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.temp_dir = tempfile.TemporaryDirectory()

        cls.test_context = fuse.context.full_chain_context(output_folder = cls.temp_dir.name)
        
        cls.test_context.set_config({"path": "/project2/lgrandi/xenonnt/simulations/testing",
                                      "file_name": "pmt_neutrons_100.root",
                                      "entry_stop": 5,
                                      })
        
        cls.run_number = "TestRun_00000"

    @classmethod
    def tearDownClass(cls):

        cls.temp_dir.cleanup()

    def tearDown(self):

        # self.temp_dir.cleanup()
        shutil.rmtree(self.temp_dir.name)
        os.makedirs(self.temp_dir.name)
    
    def test_S1PhotonHits(self):

        self.test_context.make(self.run_number, "s1_photons")

    def test_S1PhotonPropagation(self):

        self.test_context.make(self.run_number, "propagated_s1_photons")

    def test_ElectronDrift(self):

        self.test_context.make(self.run_number, "drifted_electrons")

    def test_ElectronExtraction(self):

        self.test_context.make(self.run_number, "extracted_electrons")

    def test_ElectronTiming(self):

        self.test_context.make(self.run_number, "electron_time")

    def test_SecondaryScintillation(self):

        self.test_context.make(self.run_number, "s2_photons")
        self.test_context.make(self.run_number, "s2_photons_sum")

    def test_S2PhotonPropagation(self):

        self.test_context.make(self.run_number, "propagated_s2_photons")

    def test_PMTAfterPulses(self):

        self.test_context.make(self.run_number, "pmt_afterpulses")

    def test_PMTResponseAndDAQ(self):

        self.test_context.make(self.run_number, "raw_records")
        
if __name__ == '__main__':
    unittest.main()