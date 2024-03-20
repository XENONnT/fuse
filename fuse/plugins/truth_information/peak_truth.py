import numpy as np
import strax
import straxen

from ...common import pmt_gains

export, __all__ = strax.exporter()


@export
class PeakTruth(strax.OverlapWindowPlugin):
    __version__ = "0.0.3"

    depends_on = (
        "photon_summary",
        "peak_basics",
        "microphysics_summary",
        "s1_photons",
        "s2_photons_sum",
        "drifted_electrons",
    )
    provides = "peak_truth"
    data_kind = "peaks"

    dtype = [
        ("s1_photons_in_peak", np.int32),
        ("s2_photons_in_peak", np.int32),
        ("ap_photons_in_peak", np.int32),
        ("raw_area_truth", np.float32),
        ("observable_energy_truth", np.float32),
        ("number_of_contributing_clusters_s1", np.int16),
        ("number_of_contributing_clusters_s2", np.int16),
        ("average_x_of_contributing_clusters", np.float32),
        ("average_y_of_contributing_clusters", np.float32),
        ("average_z_of_contributing_clusters", np.float32),
        ("average_x_obs_of_contributing_clusters", np.float32),
        ("average_y_obs_of_contributing_clusters", np.float32),
        ("average_z_obs_of_contributing_clusters", np.float32),
    ]
    dtype = dtype + strax.time_fields

    gain_model_mc = straxen.URLConfig(
        default="cmt://to_pe_model?version=ONLINE&run_id=plugin.run_id",
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

        photons_in_peaks = strax.split_by_containment(propagated_photons, peaks)

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

            for photon_type in photon_type_dict.keys():
                photon_cut = photons_in_peaks[i]["photon_type"] == photon_type_dict[photon_type]

                result[photon_type + "_photons_in_peak"][i] = np.sum(photon_cut)

                unique_contributing_clusters, photons_per_cluster = np.unique(
                    photons_in_peaks[i][photon_cut]["cluster_id"], return_counts=True
                )

                if photon_type == "s1":
                    result["number_of_contributing_clusters_s1"][i] = np.sum(
                        unique_contributing_clusters > 0
                    )
                    contributing_clusters_s1 = _get_cluster_information(
                        interactions_in_roi, unique_contributing_clusters
                    )
                    photons_per_cluster_s1 = photons_per_cluster
                elif photon_type == "s2":
                    result["number_of_contributing_clusters_s2"][i] = np.sum(
                        unique_contributing_clusters > 0
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

                result["raw_area_truth"][i] = np.sum(
                    photons_in_peaks[i]["photon_gain"] / self.gains[photons_in_peaks[i]["channel"]]
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
