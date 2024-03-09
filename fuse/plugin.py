import strax
import straxen
import numpy as np
import logging

logging.basicConfig(handlers=[logging.StreamHandler()])


class FuseBasePlugin(strax.Plugin):
    """Base plugin for fuse plugins."""

    # Forbid rechunking
    rechunk_on_save = False

    # Lets wait 10 minutes for the plugin to finish
    input_timeout = 600

    # Config options
    debug = straxen.URLConfig(
        default=False,
        type=bool,
        track=False,
        help="Show debug informations",
    )

    deterministic_seed = straxen.URLConfig(
        default=True,
        type=bool,
        help="Set the random seed from lineage and run_id, or pull the seed from the OS.",
    )

    def setup(self):
        super().setup()

        log = logging.getLogger(f"{self.__class__.__name__}")

        if self.debug:
            log.setLevel("DEBUG")
            log.debug(f"Running {self.__class__.__name__} version {self.__version__} in debug mode")
        else:
            log.setLevel("INFO")

        if self.deterministic_seed:
            hash_string = strax.deterministic_hash((self.run_id, self.lineage))
            self.seed = int(hash_string.encode().hex(), 16)
            self.rng = np.random.default_rng(seed=self.seed)
            log.debug(f"Generating random numbers from seed {self.seed}")
        else:
            self.rng = np.random.default_rng()
            log.debug("Generating random numbers with seed pulled from OS")


class FuseBaseDownChunkingPlugin(strax.DownChunkingPlugin, FuseBasePlugin):
    """Base plugin for fuse DownChunkingPlugins."""
