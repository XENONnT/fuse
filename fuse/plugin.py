import strax
import straxen
import numpy as np
import logging

logging.basicConfig(handlers=[logging.StreamHandler()])


class FuseBasePlugin(strax.Plugin):
    """Base plugin for fuse plugins."""

    # Forbid rechunking
    rechunk_on_save = False

    # Lets wait 15 minutes for the plugin to finish.. PI takes a while to run
    input_timeout = 900

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
        help="Set the random seed from lineage and run_id, or pull the seed from the OS",
    )

    user_defined_random_seed = straxen.URLConfig(
        default=None,
        help="Define the random seed manually. You need to set deterministic_seed to False",
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

            if self.user_defined_random_seed is not None:
                log.warning(
                    "deterministic_seed is set to True. "
                    "The provided user_defined_random_seed will not be used!"
                )

            hash_string = strax.deterministic_hash((self.run_id, self.lineage))
            self.seed = int(hash_string.encode().hex(), 16)
            self.rng = np.random.default_rng(seed=self.seed)
            log.debug(f"Generating random numbers from deterministic seed {self.seed}")
        else:

            if self.user_defined_random_seed is not None:

                assert (
                    isinstance(self.user_defined_random_seed, int)
                    and self.user_defined_random_seed > 0
                ), "user_defined_random_seed must be a positive integer!"

                self.seed = self.user_defined_random_seed
                self.rng = np.random.default_rng(self.user_defined_random_seed)
                log.info(
                    "Generating random numbers with user"
                    f"defined seed {self.user_defined_random_seed}"
                )
            else:
                self.rng = np.random.default_rng()
                log.debug("Generating random numbers with seed pulled from OS")


class FuseBaseDownChunkingPlugin(strax.DownChunkingPlugin, FuseBasePlugin):
    """Base plugin for fuse DownChunkingPlugins."""


# Modified from Cutax cutlist
class FuseCutList(strax.MergeOnlyPlugin):
    """Add nice documentation here."""

    __version__ = "0.0.0"

    save_when = strax.SaveWhen.TARGET
    cuts = ()
    # need to declare depends_on here to satisfy strax
    # https://github.com/AxFoundation/strax/blob/df18c9cef38ea1cee9737d56b1bea078ebb246a9/strax/plugin.py#L99
    depends_on = ()
    _depends_on = ()

    def setup(self):
        # I can get the conditions in the volumes from the plugin lineage!

        selection_the_plugin_depends_on = [p for p in self.depends_on if "_selection" in p]
        self.volume_names = [p[:-10] for p in selection_the_plugin_depends_on]

        self.density_dict = {}
        self.create_s2_dict = {}
        for volume in self.volume_names:
            self.density_dict[volume] = self.lineage[f"{volume}_selection"][2][
                f"xenon_density_{volume}"
            ]
            self.create_s2_dict[volume] = self.lineage[f"{volume}_selection"][2][
                f"create_S2_xenonnt_{volume}"
            ]

    def infer_dtype(self):
        dtype = super().infer_dtype()

        dtype += [
            (
                (
                    f"Combined boolean of all cuts and selections in {self.accumulated_cuts_string}",
                    self.accumulated_cuts_string,
                ),
                np.bool_,
            ),
            (
                (
                    "Flag indicating if a cluster can create a S2 signal",
                    "create_S2",
                ),
                np.bool_,
            ),
            (
                (
                    "Xenon density at the cluster position",
                    "xe_density",
                ),
                np.float32,
            ),
        ]
        return dtype

    def compute(self, **kwargs):
        cuts = super().compute(**kwargs)

        cuts_joint = np.zeros(len(cuts), self.dtype)
        strax.copy_to_buffer(
            cuts, cuts_joint, f"_copy_cuts_{strax.deterministic_hash(self.depends_on)}"
        )
        cuts_joint[self.accumulated_cuts_string] = get_accumulated_bool(cuts)

        for volume in self.volume_names:
            cuts_joint["create_S2"] = np.where(
                cuts[f"{volume}_selection"], self.create_s2_dict[volume], cuts_joint["create_S2"]
            )
            cuts_joint["xe_density"] = np.where(
                cuts[f"{volume}_selection"], self.density_dict[volume], cuts_joint["xe_density"]
            )

        return cuts_joint

    @property
    def depends_on(self):
        if not len(self._depends_on):
            deps = []
            for c in self.cuts:
                deps.extend(strax.to_str_tuple(c.provides))
            self._depends_on = tuple(deps)
        return self._depends_on

    @depends_on.setter
    def depends_on(self, str_or_tuple):
        self._depends_on = strax.to_str_tuple(str_or_tuple)


def get_accumulated_bool(array):
    """Computes accumulated boolean over all cuts and selections.

    :param array: Array containing merged cuts.
    """
    fields = array.dtype.names
    fields = np.array([f for f in fields if f not in ("time", "endtime")])

    res = np.zeros(len(array), np.bool_)
    # Modified from the default code
    for field in fields:
        if field.endswith("_selection"):
            res |= array[field]

    for field in fields:
        if field.endswith("_cut"):
            res &= array[field]

    return res
