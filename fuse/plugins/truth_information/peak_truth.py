import numpy as np
import strax
import straxen

from ...common import pmt_gains

export, __all__ = strax.exporter()


@export
class PeakTruth(strax.OverlapWindowPlugin):
    __version__ = "0.0.6"

    depends_on = (
        "photon_summary",
        "peak_basics",
        "merged_microphysics_summary",
        "merged_s1_photon_hits",
        "merged_s2_photons_sum",
        "merged_drifted_electrons",
    )
    provides = "peak_truth"
    data_kind = "peaks"

    dtype = [
        (("Number of photons from S1 scintillation in the peak.", "s1_photons_in_peak"), np.int32),
        (("Number of photons from S2 scintillation in the peak.", "s2_photons_in_peak"), np.int32),
        (("Number of photons from PMT afterpulses in the peak.", "ap_photons_in_peak"), np.int32),
        (("Number of photons from photoionization in the peak.", "pi_photons_in_peak"), np.int32),
        (
            (
                "Number of photoelectrons from S1 scintillation in the peak.",
                "s1_photoelectrons_in_peak",
            ),
            np.int32,
        ),
        (
            (
                "Number of photoelectrons from S2 scintillation in the peak.",
                "s2_photoelectrons_in_peak",
            ),
            np.int32,
        ),
        (
            (
                "Number of photoelectrons from PMT afterpulses in the peak.",
                "ap_photoelectrons_in_peak",
            ),
            np.int32,
        ),
        (
            (
                "Number of photoelectrons from photoionization in the peak.",
                "pi_photoelectrons_in_peak",
            ),
            np.int32,
        ),
        ("raw_area_truth", np.float32),
        ("observable_energy_truth", np.float32),
        ("number_of_contributing_clusters_s1", np.int16),
        ("number_of_contributing_clusters_s2", np.int16),
        ("number_of_contributing_delayed_electrons", np.int16),
        ("average_x_of_contributing_clusters", np.float32),
        ("average_y_of_contributing_clusters", np.float32),
        ("average_z_of_contributing_clusters", np.float32),
        ("average_x_obs_of_contributing_clusters", np.float32),
        ("average_y_obs_of_contributing_clusters", np.float32),
        ("average_z_obs_of_contributing_clusters", np.float32),
    ] + strax.time_fields

    gain_model_mc = straxen.URLConfig(
        default=(
            "list-to-array://xedocs://pmt_area_to_pes"
            "?as_list=True&sort=pmt&detector=tpc"
            "&run_id=plugin.run_id&version=ONLINE&attr=value"
        ),
        infer_type=False,
        help="PMT gain model",
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

    max_drift_length = straxen.URLConfig(
        default=straxen.tpc_z,
        type=(int, float),
        help="Total length of the TPC from the bottom of gate to the top of cathode wires [cm]",
    )

    drift_velocity_liquid = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=drift_velocity_liquid",
        type=(int, float),
        cache=True,
        help="Drift velocity of electrons in the liquid xenon [cm/ns]",
    )

    def setup(self):
        super().setup()

        self.gains = pmt_gains(
            self.gain_model_mc,
            digitizer_voltage_range=self.digitizer_voltage_range,
            digitizer_bits=self.digitizer_bits,
            pmt_circuit_load_resistor=self.pmt_circuit_load_resistor,
        )

    def get_window_size(self):
        drift_time_max = int(self.max_drift_length / self.drift_velocity_liquid)

        return drift_time_max * 20

    def compute(self, interactions_in_roi, propagated_photons, peaks):
        n_peaks = len(peaks)

        result = np.zeros(n_peaks, dtype=self.dtype)
        result["time"] = peaks["time"]
        result["endtime"] = peaks["endtime"]

        photons_in_peak = strax.split_by_containment(propagated_photons, peaks)

        photon_type_dict = {
            "s1": 1,
            "s2": 2,
            "ap": 0,
        }

        for i in range(n_peaks):
            contributing_clusters_s1 = np.zeros(0, dtype=interactions_in_roi.dtype)
            contributing_clusters_s2 = np.zeros(0, dtype=interactions_in_roi.dtype)
            photons_per_cluster_s1 = np.zeros(0, dtype=int)
            photons_per_cluster_s2 = np.zeros(0, dtype=int)

            photons = photons_in_peak[i]

            for photon_type in photon_type_dict.keys():
                is_from_type = photons["photon_type"] == photon_type_dict[photon_type]
                is_from_pi = (photons["cluster_id"] < 0) & (photons["photon_type"] == 2)
                has_dpe = photons["dpe"]

                # For S1 S2 AP photons in peak, we want to exclude PI photons and PEs
                # This is because we want to treat the PI as part of the bias
                result[photon_type + "_photons_in_peak"][i] = np.sum(~is_from_pi & is_from_type)
                result[photon_type + "_photoelectrons_in_peak"][i] = result[
                    photon_type + "_photons_in_peak"
                ][i]
                result[photon_type + "_photoelectrons_in_peak"][i] += np.sum(
                    ~is_from_pi & is_from_type & has_dpe
                )

                # For PI photons they are generated following S2s.
                if photon_type == "s2":
                    result["pi_photons_in_peak"][i] = np.sum(is_from_pi)
                    result["pi_photoelectrons_in_peak"][i] = (
                        np.sum(is_from_pi & has_dpe) + result["pi_photons_in_peak"][i]
                    )

                unique_contributing_clusters, photons_per_cluster = np.unique(
                    photons[is_from_type]["cluster_id"], return_counts=True
                )
                if photon_type == "s1":
                    result["number_of_contributing_clusters_s1"][i] = np.sum(
                        unique_contributing_clusters != 0
                    )
                    contributing_clusters_s1 = _get_cluster_information(
                        interactions_in_roi, unique_contributing_clusters
                    )
                    photons_per_cluster_s1 = photons_per_cluster
                elif photon_type == "s2":
                    result["number_of_contributing_clusters_s2"][i] = np.sum(
                        unique_contributing_clusters > 0
                    )
                    result["number_of_contributing_delayed_electrons"][i] = np.sum(
                        unique_contributing_clusters < 0
                    )
                    contributing_clusters_s2 = _get_cluster_information(
                        interactions_in_roi, unique_contributing_clusters
                    )
                    photons_per_cluster_s2 = photons_per_cluster

            if (result["s1_photons_in_peak"][i] + result["s2_photons_in_peak"][i]) > 0:
                positions_to_evaluate = ["x", "y", "z", "x_obs", "y_obs", "z_obs"]

                for position in positions_to_evaluate:
                    result_name = "average_" + position + "_of_contributing_clusters"

                    result[result_name][i] = weighted_position_average(
                        position,
                        contributing_clusters_s1,
                        contributing_clusters_s2,
                        photons_per_cluster_s1,
                        photons_per_cluster_s2,
                    )

                # Assume that we calibrate or detector
                # so that sum_s2_photons would give us the observed energy
                energy_of_s2_photons_in_peak = (
                    photons_per_cluster_s2
                    / contributing_clusters_s2["sum_s2_photons"]
                    * contributing_clusters_s2["ed"]
                )
                # Same for S1 but with n_s1_photon_hits
                energy_of_s1_photons_in_peak = (
                    photons_per_cluster_s1
                    / contributing_clusters_s1["n_s1_photon_hits"]
                    * contributing_clusters_s1["ed"]
                )
                # Sum up up the two:
                result["observable_energy_truth"][i] = np.sum(
                    energy_of_s1_photons_in_peak
                ) + np.sum(energy_of_s2_photons_in_peak)

                # Calculate the raw area truth
                # exclude PMT AP photons as well as photons from delayed electrons
                masked_photons = photons_in_peak[i]
                masked_photons = masked_photons[
                    (masked_photons["photon_type"] != 0) & (masked_photons["cluster_id"] > 0)
                ]

                result["raw_area_truth"][i] = np.sum(
                    masked_photons["photon_gain"] / self.gains[masked_photons["channel"]]
                )
        return result


def _get_cluster_information(interactions_in_roi, unique_contributing_clusters):
    contributing_cluster_informations = interactions_in_roi[
        np.isin(interactions_in_roi["cluster_id"], unique_contributing_clusters)
    ]
    sort_index = np.argsort(contributing_cluster_informations["cluster_id"])
    return contributing_cluster_informations[sort_index]


def weighted_position_average(coord, contr_s1_cl, contr_s2_cl, ph_per_cl_s1, ph_per_cl_s2):
    return np.sum(
        np.concatenate([contr_s1_cl[coord] * ph_per_cl_s1, contr_s2_cl[coord] * ph_per_cl_s2])
    ) / np.sum(np.concatenate([ph_per_cl_s1, ph_per_cl_s2]))
