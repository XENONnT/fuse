import unittest
import fuse
import tempfile

from straxen import URLConfig

class TestFullChainwDelayedElectrons(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.temp_dir = tempfile.TemporaryDirectory()

        url_string = 'simple_load://resource://format://fax_config_nt_sr0_v4.json?&fmt=json'
        config = URLConfig.evaluate_dry(url_string) 

        self.test_context = fuse.context.full_chain_w_photo_ionization_context(
            self.temp_dir.name,
            config = config,
            )

        self.test_context.set_config({"path": "/project2/lgrandi/xenonnt/simulations/testing",
                                      "file_name": "pmt_neutrons_100.root",
                                      "entry_stop": 3,
                                      })
        
        self.run_number = "TestRun_00000"

    @classmethod
    def tearDownClass(self):

        self.temp_dir.cleanup()
    
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
    
    def test_PhotoIonizationElectrons(self):

        self.test_context.make(self.run_number, "photo_ionization_electrons")

    def test_DelayedElectronsDrift(self):

        self.test_context.make(self.run_number, "drifted_delayed_electrons")

    def test_DelayedElectronsExtraction(self):

        self.test_context.make(self.run_number, "extracted_delayed_electrons")

    def test_DelayedElectronsTiming(self):

        self.test_context.make(self.run_number, "delayed_electrons_time")

    def test_DelayedElectronsSecondaryScintillation(self):

        self.test_context.make(self.run_number, "delayed_electrons_s2_photons")
        self.test_context.make(self.run_number, "delayed_electrons_s2_photons_sum")

    def test_S2PhotonPropagation(self):

        self.test_context.make(self.run_number, "propagated_s2_photons")

    def test_PMTAfterPulses(self):

        self.test_context.make(self.run_number, "pmt_afterpulses")

    def test_PMTResponseAndDAQ(self):

        self.test_context.make(self.run_number, "raw_records")
        
if __name__ == '__main__':
    unittest.main()