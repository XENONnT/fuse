import strax
import straxen

import numpy as np

from ...common import pmt_gains
from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()


@export
class S1PhotonHits(FuseBasePlugin):
    """Plugin to simulate the number of detected S1 photons using a S1 light
    collection efficiency map."""

    __version__ = "0.2.2"

    depends_on = "microphysics_summary"
    provides = "s1_photon_hits"
    data_kind = "interactions_in_roi"

    save_when = strax.SaveWhen.ALWAYS

    dtype = [
        (("Number detected S1 photons", "n_s1_photon_hits"), np.int32),
    ] + strax.time_fields

    # Config options
    pmt_circuit_load_resistor = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=pmt_circuit_load_resistor",
        type=(int, float),
        cache=True,
        help="PMT circuit load resistor [kg m^2/(s^3 A)]",
    )

    digitizer_bits = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?&fmt=json&take=digitizer_bits",
        type=(int, float),
        cache=True,
        help="Number of bits of the digitizer boards",
    )

    digitizer_voltage_range = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=digitizer_voltage_range",
        type=(int, float),
        cache=True,
        help="Voltage range of the digitizer boards [V]",
    )

    gain_model_mc = straxen.URLConfig(
        default=(
            "list-to-array://xedocs://pmt_area_to_pes"
            "?as_list=True&sort=pmt&detector=tpc"
            "&run_id=plugin.run_id&version=ONLINE&attr=value"
        ),
        infer_type=False,
        help="PMT gain model",
    )

    s1_lce_correction_map = straxen.URLConfig(
        default="lce_from_pattern_map://pattern_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=s1_pattern_map"
        "&fmt=pkl"
        "&pmt_mask=plugin.pmt_mask",
        cache=True,
        help="S1 LCE correction map",
    )

    p_double_pe_emision = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=p_double_pe_emision",
        type=(int, float),
        cache=True,
        help="Probability of double photo-electron emission",
    )

    s1_detection_efficiency = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=s1_detection_efficiency",
        type=(int, float),
        cache=True,
        help="S1 detection efficiency",
    )

    def setup(self):
        super().setup()

        self.gains = pmt_gains(
            self.gain_model_mc,
            digitizer_voltage_range=self.digitizer_voltage_range,
            digitizer_bits=self.digitizer_bits,
            pmt_circuit_load_resistor=self.pmt_circuit_load_resistor,
        )

        self.pmt_mask = np.array(self.gains) > 0  # Converted from to pe (from xedocs by default)

    def compute(self, interactions_in_roi):
        # Just apply this to clusters with photons
        mask = interactions_in_roi["photons"] > 0

        if len(interactions_in_roi[mask]) == 0:
            empty_result = np.zeros(len(interactions_in_roi), self.dtype)
            empty_result["time"] = interactions_in_roi["time"]
            empty_result["endtime"] = interactions_in_roi["endtime"]
            return empty_result

        x = interactions_in_roi[mask]["x"]
        y = interactions_in_roi[mask]["y"]
        z = interactions_in_roi[mask]["z"]
        n_photons = interactions_in_roi[mask]["photons"].astype(np.int64)

        positions = np.array([x, y, z]).T

        n_photon_hits = self.get_n_photons(
            n_photons=n_photons,
            positions=positions,
        )

        result = np.zeros(interactions_in_roi.shape[0], dtype=self.dtype)

        result["time"] = interactions_in_roi["time"]
        result["endtime"] = interactions_in_roi["endtime"]

        result["n_s1_photon_hits"][mask] = n_photon_hits

        return result

    def get_n_photons(self, n_photons, positions):
        """Calculates number of detected photons based on number of photons in
        total and the positions.

        Args:
            n_photons: 1d array of ints with number of emitted S1 photons
            positions: 2d array with xyz positions of interactions
            s1_lce_correction_map: interpolator instance of s1 light yield map
            config: dict wfsim config
        Returns:
            return array with number photons
        """
        ly = self.s1_lce_correction_map(positions)
        # Depending on if you use the data driven or mc pattern map for light yield
        # the shape of n_photon_hits will change. Mc needs a squeeze
        if len(ly.shape) != 1:
            ly = np.squeeze(ly, axis=-1)
        ly /= 1 + self.p_double_pe_emision
        ly *= self.s1_detection_efficiency

        n_photon_hits = self.rng.binomial(n=n_photons, p=ly)

        return n_photon_hits
