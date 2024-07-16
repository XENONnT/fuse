from immutabledict import immutabledict

import numpy as np
import strax
import straxen

from ...common import pmt_gains
from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()


@export
class SecondaryScintillation(FuseBasePlugin):
    """Plugin to simulate the secondary scintillation process in the gas
    phase."""

    __version__ = "0.2.0"

    result_name_photons = "s2_photons"
    result_name_photons_sum = "s2_photons_sum"

    depends_on = (
        "microphysics_summary",
        "drifted_electrons",
        "extracted_electrons",
        "electron_time",
    )
    provides = (result_name_photons, result_name_photons_sum)
    data_kind = {
        result_name_photons: "individual_electrons",
        result_name_photons_sum: "interactions_in_roi",
    }

    dtype_photons = [
        (("Number of photons produced by the extracted electron", "n_s2_photons"), np.int32),
    ] + strax.time_fields
    dtype_sum_photons = [
        (
            (
                "Sum of all photons produced by electrons originating from the same cluster",
                "sum_s2_photons",
            ),
            np.int32,
        ),
    ] + strax.time_fields

    dtype = dict()
    dtype[result_name_photons] = dtype_photons
    dtype[result_name_photons_sum] = dtype_sum_photons

    save_when = immutabledict(
        {result_name_photons: strax.SaveWhen.TARGET, result_name_photons_sum: strax.SaveWhen.ALWAYS}
    )

    # Config options
    s2_secondary_sc_gain_mc = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=s2_secondary_sc_gain",
        type=(int, float),
        cache=True,
        help="Secondary scintillation gain [PE/e-]",
    )

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

    se_gain_from_map = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=se_gain_from_map",
        cache=True,
        help="Boolean indication if the secondary scintillation gain is taken from a map",
    )

    p_double_pe_emision = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=p_double_pe_emision",
        type=(int, float),
        cache=True,
        help="Probability of double photo-electron emission",
    )

    se_gain_map = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=se_gain_map"
        "&fmt=json",
        cache=True,
        help="Map of the single electron gain",
    )

    s2_correction_map = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=s2_correction_map"
        "&fmt=json",
        cache=True,
        help="S2 correction map",
    )

    gain_model_mc = straxen.URLConfig(
        default="cmt://to_pe_model?version=ONLINE&run_id=plugin.run_id",
        infer_type=False,
        help="PMT gain model",
    )

    n_top_pmts = straxen.URLConfig(
        type=int,
        help="Number of PMTs on top array",
    )

    n_tpc_pmts = straxen.URLConfig(
        type=int,
        help="Number of PMTs in the TPC",
    )

    s2_mean_area_fraction_top = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=s2_mean_area_fraction_top",
        type=(int, float),
        cache=True,
        help="Mean S2 area fraction top",
    )

    s2_pattern_map = straxen.URLConfig(
        default="s2_aft_scaling://pattern_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=s2_pattern_map"
        "&fmt=pkl"
        "&pmt_mask=plugin.pmt_mask"
        "&s2_mean_area_fraction_top=plugin.s2_mean_area_fraction_top"
        "&n_tpc_pmts=plugin.n_tpc_pmts"
        "&n_top_pmts=plugin.n_top_pmts"
        "&turned_off_pmts=plugin.turned_off_pmts",
        cache=True,
        help="S2 pattern map",
    )

    def setup(self):
        super().setup()

        self.gains = pmt_gains(
            self.gain_model_mc,
            digitizer_voltage_range=self.digitizer_voltage_range,
            digitizer_bits=self.digitizer_bits,
            pmt_circuit_load_resistor=self.pmt_circuit_load_resistor,
        )

        self.pmt_mask = np.array(self.gains)

    def compute(self, interactions_in_roi, individual_electrons):
        # Just apply this to clusters with electrons
        mask = interactions_in_roi["n_electron_extracted"] > 0

        if len(interactions_in_roi[mask]) == 0:
            empty_result = np.zeros(
                len(interactions_in_roi), self.dtype[self.result_name_photons_sum]
            )
            empty_result["time"] = interactions_in_roi["time"]
            empty_result["endtime"] = interactions_in_roi["endtime"]

            return {
                self.result_name_photons: np.zeros(0, self.dtype[self.result_name_photons]),
                self.result_name_photons_sum: empty_result,
            }

        positions = np.array([individual_electrons["x"], individual_electrons["y"]]).T

        electron_gains = self.get_s2_light_yield(positions=positions)

        n_photons_per_ele = self.rng.poisson(electron_gains)

        result_photons = np.zeros(
            len(n_photons_per_ele), dtype=self.dtype[self.result_name_photons]
        )
        result_photons["n_s2_photons"] = n_photons_per_ele
        result_photons["time"] = individual_electrons["time"]
        result_photons["endtime"] = individual_electrons["endtime"]

        # Calculate the sum of photons per interaction
        grouped_result_photons, unique_cluster_id = group_result_photons_by_cluster_id(
            result_photons, individual_electrons["cluster_id"]
        )
        sum_photons_per_interaction = np.array(
            [np.sum(element["n_s2_photons"]) for element in grouped_result_photons]
        )

        # Bring sum_photons_per_interaction into the same cluster order as interactions_in_roi
        # Maybe this line is too complicated...
        sum_photons_per_interaction_reordered = [
            sum_photons_per_interaction[np.argwhere(unique_cluster_id == element)[0][0]]
            for element in interactions_in_roi["cluster_id"][mask]
        ]

        result_sum_photons = np.zeros(
            len(interactions_in_roi), dtype=self.dtype[self.result_name_photons_sum]
        )
        result_sum_photons["sum_s2_photons"][mask] = sum_photons_per_interaction_reordered
        result_sum_photons["time"] = interactions_in_roi["time"]
        result_sum_photons["endtime"] = interactions_in_roi["endtime"]

        return {
            self.result_name_photons: strax.sort_by_time(result_photons),
            self.result_name_photons_sum: result_sum_photons,
        }

    def get_s2_light_yield(self, positions):
        """Calculate s2 light yield...

        Args:
            positions: 2d array of positions (floats) returns array
                of floats (mean expectation)
        """

        if self.se_gain_from_map:
            sc_gain = self.se_gain_map(positions)
        else:
            # calculate it from MC pattern map directly if no "se_gain_map" is given
            sc_gain = self.s2_correction_map(positions)
            sc_gain *= self.s2_secondary_sc_gain_mc

        # Depending on if you use the data driven or mc pattern map for light yield for S2
        # The shape of n_photon_hits will change. Mc needs a squeeze
        if len(sc_gain.shape) != 1:
            sc_gain = np.squeeze(sc_gain, axis=-1)

        # sc gain should has the unit of pe / electron,
        # Here we divide 1 + dpe to get nphoton / electron
        sc_gain /= 1 + self.p_double_pe_emision

        # Data driven map contains nan, will be set to 0 here
        sc_gain[np.isnan(sc_gain)] = 0

        return sc_gain


def group_result_photons_by_cluster_id(result, cluster_id):
    """Function to group result_photons by cluster_id."""

    sort_index = np.argsort(cluster_id)

    cluster_id_sorted = cluster_id[sort_index]
    result_sorted = result[sort_index]

    unique_cluster_id, split_position = np.unique(cluster_id_sorted, return_index=True)
    return np.split(result_sorted, split_position[1:]), unique_cluster_id
