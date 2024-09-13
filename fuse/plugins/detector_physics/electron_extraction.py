import numpy as np
import strax
import straxen

from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()

@export
class ElectronExtraction(FuseBasePlugin):
    """Plugin to simulate the loss of electrons during the extraction of
    drifted electrons from the liquid into the gas phase."""

    __version__ = "0.3.0"

    depends_on = ("electrons_at_interface")
    provides = "extracted_electrons"
    data_kind = "individual_electrons"

    save_when = strax.SaveWhen.TARGET

    dtype = [
        (("x position of the electron at the interface [cm]", "x_interface"), np.float32),
        (("y position of the electron at the interface [cm]", "y_interface"), np.float32),
        (("ID of the cluster creating the electron", "cluster_id"), np.int32),
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

    electron_trapping_time = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=electron_trapping_time",
        type=(int, float),
        cache=True,
        help="Time scale electrons are trapped at the liquid gas interface",
    )

    def compute(self, individual_electrons):

        position = np.array(
            [individual_electrons['x_interface'], individual_electrons['y_interface']]
        ).T

        if self.ext_eff_from_map:
            # Extraction efficiency is g2(x,y)/SE_gain(x,y)
            rel_s2_cor = self.s2_correction_map(position)
            # Doesn't always need to be flattened, but if s2_correction_map = False,
            # then map is made from MC
            rel_s2_cor = rel_s2_cor.flatten()

            if self.se_gain_from_map:
                se_gains = self.se_gain_map(position)
            else:
                # Is in get_s2_light_yield map is scaled according to relative s2 correction
                # We also need to do it here to have consistent g2
                se_gains = rel_s2_cor * self.s2_secondary_sc_gain_mc
            cy = self.g2_mean * rel_s2_cor / se_gains
        else:
            cy = self.electron_extraction_yield

        extraction_mask = self.rng.binomial(1, p=cy, size = position.shape[0]).astype(bool)

        result = np.zeros(np.sum(extraction_mask), dtype=self.dtype)
        result["time"] = self.extraction_delay(individual_electrons[extraction_mask]["time"])
        result["endtime"] = result["time"]
        result["x_interface"] = individual_electrons[extraction_mask]["x_interface"]
        result["y_interface"] = individual_electrons[extraction_mask]["y_interface"]
        result["cluster_id"] = individual_electrons[extraction_mask]["cluster_id"]

        return result

    def extraction_delay(
        self,
        electron_times,
    ):
        timing = self.rng.exponential(self.electron_trapping_time, size=electron_times.shape[0])

        return electron_times + timing.astype(np.int64)
