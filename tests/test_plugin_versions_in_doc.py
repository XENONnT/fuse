import os
import unittest
import tempfile
import timeout_decorator
import fuse

TIMEOUT = 240


def read_version_from_documentation(plugin_name, path_to_micrphysics_doc):
    doc_version = None

    with open(f"{path_to_micrphysics_doc}/{plugin_name}.rst", "r") as file:
        for line in file:
            if "__version__" in line:
                doc_version = line.split("=")[1].strip('"\n ').lstrip()
                break

    if not doc_version:
        raise ValueError(f"Could not find version for plugin {plugin_name} in the documentation!")

    return doc_version


microphysics_name_dict = {
    "ChunkInput": "geant4_interactions",
    "FindCluster": "cluster_index",
    "MergeCluster": "clustered_interactions",
    "ElectricField": "electric_field_values",
    "VolumesMerger": "interactions_in_roi",
    "XENONnT_TPC": "tpc_interactions",
    "XENONnT_BelowCathode": "below_cathode_interactions",
    "MicroPhysicsSummary": "microphysics_summary",
    "NestYields": "quanta",
}

detector_physics_name_dict = {
    "ElectronDrift": "drifted_electrons",
    "ElectronExtraction": "extracted_electrons",
    "ElectronTiming": "electron_time",
    "SecondaryScintillation": "s2_photons",
    "S1PhotonHits": "s1_photons",
    "S2PhotonPropagation": "propagated_s2_photons",
    "S1PhotonPropagation": "propagated_s1_photons",
}

pmt_and_daq_name_dict = {
    "PulseWindow": "pulse_windows",
    "PhotonSummary": "photon_summary",
    "PMTAfterPulses": "pmt_afterpulses",
    "PMTResponseAndDAQ": "raw_records",
}


class TestPluginVersionsInDocumentation(unittest.TestCase):
    """Test clas to check if the plugin version in the documentation are the
    same as the in the code."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()

        test_context = fuse.context.full_chain_context(
            output_folder=cls.temp_dir.name, run_without_proper_corrections=True
        )
        cls.plugin_registry = test_context._plugin_class_registry

        cls.this_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    @timeout_decorator.timeout(TIMEOUT, exception_message="Microphysics version tests timed out")
    def test_microphysics_versions(self):
        for plugin_name, plugin_provides in microphysics_name_dict.items():
            plugin_version = self.plugin_registry[plugin_provides].__version__

            path_to_micrphysics_doc = os.path.join(
                self.this_dir, "docs", "source", "plugins", "micro_physics"
            )
            doc_version = read_version_from_documentation(plugin_name, path_to_micrphysics_doc)

            msg = (
                f"Plugin {plugin_name} version in the documentation is {doc_version} "
                f"and in the code is {plugin_version}! Remember to update the documentation!"
            )
            self.assertEqual(
                plugin_version,
                doc_version,
                msg=msg,
            )

    @timeout_decorator.timeout(
        TIMEOUT, exception_message="Detector physics version tests timed out"
    )
    def test_detector_physics_versions(self):
        for plugin_name, plugin_provides in detector_physics_name_dict.items():
            plugin_version = self.plugin_registry[plugin_provides].__version__

            path_to_detector_physics_doc = os.path.join(
                self.this_dir, "docs", "source", "plugins", "detector_physics"
            )
            doc_version = read_version_from_documentation(plugin_name, path_to_detector_physics_doc)

            msg = (
                f"Plugin {plugin_name} version in the documentation is {doc_version} "
                f"and in the code is {plugin_version}! Remember to update the documentation!"
            )
            self.assertEqual(
                plugin_version,
                doc_version,
                msg=msg,
            )

    @timeout_decorator.timeout(TIMEOUT, exception_message="PMT response version tests timed out")
    def test_pmt_response_versions(self):
        for plugin_name, plugin_provides in pmt_and_daq_name_dict.items():
            plugin_version = self.plugin_registry[plugin_provides].__version__

            path_to_pmt_response_doc = os.path.join(
                self.this_dir, "docs", "source", "plugins", "pmt_and_daq"
            )
            doc_version = read_version_from_documentation(plugin_name, path_to_pmt_response_doc)

            msg = (
                f"Plugin {plugin_name} version in the documentation is {doc_version} "
                f"and in the code is {plugin_version}! Remember to update the documentation!"
            )
            self.assertEqual(
                plugin_version,
                doc_version,
                msg=msg,
            )
