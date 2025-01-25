import os
import shutil
import unittest
import tempfile
import timeout_decorator
import fuse
import utilix
from _utils import test_root_file_name

TIMEOUT = 60


class TestPluginRandomSeeds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.run_number = "TestRun_00000"

        cls.test_context = fuse.context.full_chain_context(
            cls.temp_dir.name, run_without_proper_corrections=True
        )
        cls.test_context.set_config(
            {
                "path": cls.temp_dir.name,
                "file_name": test_root_file_name,
                "entry_start": 0,
                "entry_stop": 10,
            }
        )

        # Get all registered fuse plutins.
        cls.all_registered_fuse_plugins = {}
        for key, value in cls.test_context._plugin_class_registry.items():
            if "fuse" in str(value):
                cls.all_registered_fuse_plugins[key] = value

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

    @timeout_decorator.timeout(TIMEOUT, exception_message="test_if_plugins_get_user_seed timed out")
    def test_if_plugins_get_user_seed(self):
        self.test_context.set_config(
            {
                "deterministic_seed": False,
                "user_defined_random_seed": 42,
            }
        )

        # Lets check the random seed for all fuse plugins
        for key in self.all_registered_fuse_plugins.keys():
            plugin = self.test_context.get_single_plugin(self.run_number, key)

            if hasattr(plugin, "seed"):
                assert (
                    plugin.seed == 42
                ), f"Expecting seed to be 42, but got {plugin.seed} for {key} plugin"

    @timeout_decorator.timeout(
        TIMEOUT, exception_message="test_if_plugins_with_rng_have_a_proper_seed timed out"
    )
    def test_if_plugins_with_rng_have_a_proper_seed(self):

        # Lets check the random seed for all fuse plugins
        for key in self.all_registered_fuse_plugins.keys():
            plugin = self.test_context.get_single_plugin(self.run_number, key)

            if hasattr(plugin, "rng"):
                if not hasattr(plugin, "seed"):
                    raise ValueError(f"Plugin {key} has rng but no seed")

    @timeout_decorator.timeout(
        TIMEOUT, exception_message="test_if_negative_seeds_are_intercepted timed out"
    )
    def test_if_negative_seeds_are_intercepted(self):
        self.test_context.set_config(
            {
                "deterministic_seed": False,
                "user_defined_random_seed": -42,
            }
        )

        # Lets check the random seed for all fuse plugins
        for key in self.all_registered_fuse_plugins.keys():

            with self.assertRaises(AssertionError):
                plugin = self.test_context.get_single_plugin(self.run_number, key)

                # Some plugins have no seed, so we can't check for negative seeds.
                if not hasattr(plugin, "seed"):
                    raise AssertionError(f"Plugin {key} has no seed")

    @timeout_decorator.timeout(
        TIMEOUT * 2, exception_message="test_if_run_number_changes_deterministic_seed timed out"
    )
    def test_if_run_number_changes_deterministic_seed(self):

        self.test_context.set_config({"deterministic_seed": True})

        # Lets check the random seed for all fuse plugins
        for key in self.all_registered_fuse_plugins.keys():

            plugin = self.test_context.get_single_plugin("00000", key)

            if hasattr(plugin, "seed"):

                seed_0 = self.test_context.get_single_plugin("00000", key).seed
                seed_1 = self.test_context.get_single_plugin("00001", key).seed

                assert (
                    seed_0 != seed_1
                ), f"Expecting seed to be different for different run numbers for {key} plugin"

    @timeout_decorator.timeout(
        TIMEOUT, exception_message="test_if_tracked_config_changes_deterministic_seed timed out"
    )
    def test_if_tracked_config_changes_deterministic_seed(self):

        self.test_context.set_config({"deterministic_seed": True})

        seed_0 = self.test_context.get_single_plugin(self.run_number, "raw_records").seed

        # Change some tracked config argument.
        self.test_context.set_config({"entry_stop": 20})
        seed_1 = self.test_context.get_single_plugin(self.run_number, "raw_records").seed

        assert seed_0 != seed_1, "Expecting seed to be different for different config args!"
