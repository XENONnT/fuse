import numpy as np
import strax
import straxen

from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()


@export
class ElectronExtraction(FuseBasePlugin):
    """Plugin to simulate the loss of electrons during the extraction of
    drifted electrons from the liquid into the gas phase."""

    __version__ = "0.2.0"

    depends_on = ("microphysics_summary", "drifted_electrons")
    provides = "extracted_electrons"
    data_kind = "interactions_in_roi"

    save_when = strax.SaveWhen.ALWAYS

    dtype = [
        (("Number of electrons extracted into the gas phase", "n_electron_extracted"), np.int32),
    ] + strax.time_fields

    # Config options
    s2_secondary_sc_gain_mc = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=s2_secondary_sc_gain",
        type=(int, float),
        cache=True,
        help="Secondary scintillation gain [PE/e-]",
    )
    # Rename? -> g2_value in beta_yields model
    g2_mean = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?&fmt=json&take=g2_mean",
        type=(int, float),
        cache=True,
        help="Mean value of the g2 gain [PE/e-]",
    )

    electron_extraction_yield = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=electron_extraction_yield",
        type=(int, float),
        cache=True,
        help="Electron extraction yield [electron_extracted/electron]",
    )

    ext_eff_from_map = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=ext_eff_from_map",
        type=bool,
        cache=True,
        help="Boolean indication if the extraction efficiency is taken from a map",
    )

    se_gain_from_map = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=se_gain_from_map",
        type=bool,
        cache=True,
        help="Boolean indication if the secondary scintillation gain is taken from a map",
    )

    s2_correction_map = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=s2_correction_map"
        "&fmt=json",
        cache=True,
        help="S2 correction map",
    )

    se_gain_map = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=se_gain_map"
        "&fmt=json",
        cache=True,
        help="Map of the single electron gain",
    )

    def compute(self, interactions_in_roi):
        # Just apply this to clusters with photons
        mask = interactions_in_roi["electrons"] > 0

        if len(interactions_in_roi[mask]) == 0:
            empty_result = np.zeros(len(interactions_in_roi), self.dtype)
            empty_result["time"] = interactions_in_roi["time"]
            empty_result["endtime"] = interactions_in_roi["endtime"]
            return empty_result

        x = interactions_in_roi[mask]["x_obs"]
        y = interactions_in_roi[mask]["y_obs"]

        xy_int = np.array([x, y]).T  # maps are in R_true, so orginal position should be here

        if self.ext_eff_from_map:
            # Extraction efficiency is g2(x,y)/SE_gain(x,y)
            rel_s2_cor = self.s2_correction_map(xy_int)
            # Doesn't always need to be flattened, but if s2_correction_map = False,
            # then map is made from MC
            rel_s2_cor = rel_s2_cor.flatten()

            if self.se_gain_from_map:
                se_gains = self.se_gain_map(xy_int)
            else:
                # Is in get_s2_light_yield map is scaled according to relative s2 correction
                # We also need to do it here to have consistent g2
                se_gains = rel_s2_cor * self.s2_secondary_sc_gain_mc
            cy = self.g2_mean * rel_s2_cor / se_gains
        else:
            cy = self.electron_extraction_yield

        n_electron = self.rng.binomial(n=interactions_in_roi[mask]["n_electron_interface"], p=cy)

        result = np.zeros(len(interactions_in_roi), dtype=self.dtype)
        result["n_electron_extracted"][mask] = n_electron
        result["time"] = interactions_in_roi["time"]
        result["endtime"] = interactions_in_roi["endtime"]

        return result
