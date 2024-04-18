import strax
import straxen
import numpy as np
import numba

from ...common import pmt_gains

export, __all__ = strax.exporter()


@export
class RecordsTruth(strax.Plugin):
    """Plugin that computes the truth information for raw_records."""

    __version__ = "0.0.2"

    depends_on = ("photon_summary", "raw_records")
    provides = "records_truth"
    data_kind = "raw_records"

    dtype = [
        (("Number of S1 photons in record", "s1_photons_in_record"), np.int32),
        (("Number of S2 photons in record", "s2_photons_in_record"), np.int32),
        (("Number of AP photons in record", "ap_photons_in_record"), np.int32),
        (("Sum of the photon gains", "raw_area"), np.float32),
    ]

    dtype = strax.interval_dtype + dtype

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

    def setup(self):
        super().setup()

        self.gains = pmt_gains(
            self.gain_model_mc,
            digitizer_voltage_range=self.digitizer_voltage_range,
            digitizer_bits=self.digitizer_bits,
            pmt_circuit_load_resistor=self.pmt_circuit_load_resistor,
        )

    def compute(self, propagated_photons, raw_records):
        result = np.zeros(len(raw_records), dtype=self.dtype)
        result["time"] = raw_records["time"]
        result["length"] = raw_records["length"]
        result["dt"] = raw_records["dt"]
        result["channel"] = raw_records["channel"]

        photons_per_channel, unique_photon_channels = split_photons_by_channel(propagated_photons)
        records_per_channel, unique_records_channels = split_records_by_channel(raw_records)

        for channel in np.unique(raw_records["channel"]):
            records_in_channel = records_per_channel[
                np.argwhere(unique_records_channels == channel)[0][0]
            ]
            photons_in_channel = photons_per_channel[
                np.argwhere(unique_photon_channels == channel)[0][0]
            ]

            result_buffer = np.zeros(len(records_in_channel), dtype=self.dtype)

            photons_per_cluster = strax.split_by_containment(photons_in_channel, records_in_channel)

            fill_result_buffer(photons_per_cluster, result_buffer)

            result_mask = result["channel"] == channel

            result["raw_area"][result_mask] = result_buffer["raw_area"] / self.gains[channel]
            result["s1_photons_in_record"][result_mask] = result_buffer["s1_photons_in_record"]
            result["s2_photons_in_record"][result_mask] = result_buffer["s2_photons_in_record"]
            result["ap_photons_in_record"][result_mask] = result_buffer["ap_photons_in_record"]

        return result


@numba.njit()
def fill_result_buffer(list_input, result_buffer):
    for i, photons in enumerate(list_input):
        # if len(photons) > 0:
        result_buffer["raw_area"][i] = np.sum(photons["photon_gain"])
        result_buffer["s1_photons_in_record"][i] = np.sum(photons["photon_type"] == 1)
        result_buffer["s2_photons_in_record"][i] = np.sum(photons["photon_type"] == 2)
        result_buffer["ap_photons_in_record"][i] = np.sum(photons["photon_type"] == 0)


def split_photons_by_channel(propagated_photons):
    sort_index = np.argsort(propagated_photons[["channel", "time"]])

    propagated_photons_sorted = propagated_photons[sort_index]

    unique_photon_channels, split_position = np.unique(
        propagated_photons_sorted["channel"], return_index=True
    )
    return np.split(propagated_photons_sorted, split_position[1:]), unique_photon_channels


def split_records_by_channel(records):
    sort_index = np.argsort(records[["channel", "time"]])

    records_sorted = records[sort_index]

    unique_channels, split_position = np.unique(records_sorted["channel"], return_index=True)
    return np.split(records_sorted, split_position[1:]), unique_channels
