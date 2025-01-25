import strax
import numpy as np
import straxen

from ...dtypes import propagated_photons_fields
from ...common import pmt_gains
from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()


@export
class PMTAfterPulses(FuseBasePlugin):
    """Plugin to simulate PMT afterpulses using a precomputed afterpulse
    cumulative distribution function.

    In the simulation afterpulses will be saved as a list of "pseudo"
    photons. These "photons" can then be combined with real photons from
    S1 and S2 signals to create a waveform.
    """

    __version__ = "0.3.1"

    depends_on = ("propagated_s2_photons", "propagated_s1_photons")
    provides = "pmt_afterpulses"
    data_kind = "ap_photons"

    save_when = strax.SaveWhen.TARGET

    dtype = propagated_photons_fields + strax.time_fields

    # Config options

    enable_pmt_afterpulses = straxen.URLConfig(
        default=True,
        type=bool,
        track=True,
        help="Decide if you want to to enable PMT afterpulsing",
    )

    pmt_ap_t_modifier = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=pmt_ap_t_modifier",
        type=(int, float),
        cache=True,
        help="PMT afterpulse time modifier",
    )

    pmt_ap_modifier = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=pmt_ap_modifier",
        type=(int, float),
        cache=True,
        help="PMT afterpulse modifier",
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

    gain_model_mc = straxen.URLConfig(
        default=(
            "list-to-array://xedocs://pmt_area_to_pes"
            "?as_list=True&sort=pmt&detector=tpc"
            "&run_id=plugin.run_id&version=ONLINE&attr=value"
        ),
        infer_type=False,
        help="PMT gain model",
    )

    photon_ap_cdfs = straxen.URLConfig(
        default="simple_load://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=photon_ap_cdfs"
        "&fmt=json.gz",
        cache=True,
        help="Afterpuse cumulative distribution functions",
    )

    def setup(self):
        super().setup()

        self.gains = pmt_gains(
            self.gain_model_mc,
            digitizer_voltage_range=self.digitizer_voltage_range,
            digitizer_bits=self.digitizer_bits,
            pmt_circuit_load_resistor=self.pmt_circuit_load_resistor,
        )

        self.uniform_to_pmt_ap = self.photon_ap_cdfs

        for k in self.uniform_to_pmt_ap.keys():
            for q in self.uniform_to_pmt_ap[k].keys():
                if isinstance(self.uniform_to_pmt_ap[k][q], list):
                    self.uniform_to_pmt_ap[k][q] = np.array(self.uniform_to_pmt_ap[k][q])

    def compute(self, s1_photons, s2_photons):
        if not self.enable_pmt_afterpulses or (len(s1_photons) == 0 and len(s2_photons) == 0):
            return np.zeros(0, dtype=self.dtype)

        merged_photons = np.concatenate([s1_photons, s2_photons])
        s1_photons = None
        s2_photons = None

        # Sort all photons by time
        sortind = np.argsort(merged_photons["time"])
        merged_photons = merged_photons[sortind]

        _photon_timings = merged_photons["time"]
        _photon_channels = merged_photons["channel"]
        _photon_is_dpe = merged_photons["dpe"]

        ap_photon_timings, ap_photon_channels, ap_photon_gains = self.photon_afterpulse(
            _photon_timings, _photon_channels, _photon_is_dpe
        )
        ap_photon_is_dpe = np.zeros_like(ap_photon_timings).astype(np.bool_)

        result = np.zeros(len(ap_photon_channels), dtype=self.dtype)
        result["channel"] = ap_photon_channels
        result["time"] = ap_photon_timings
        result["endtime"] = ap_photon_timings
        result["dpe"] = ap_photon_is_dpe
        result["photon_gain"] = ap_photon_gains

        result["cluster_id"] = -1 * np.ones(len(ap_photon_channels))

        result = strax.sort_by_time(result)

        return result

    def photon_afterpulse(
        self, merged_photon_timings, merged_photon_channels, merged_photon_id_dpe
    ):
        """For pmt afterpulses, gain and dpe generation is a bit different from
        standard photons."""
        element_list = self.uniform_to_pmt_ap.keys()
        _photon_timings = []
        _photon_channels = []
        _photon_amplitude = []

        for element in element_list:
            delaytime_cdf = self.uniform_to_pmt_ap[element]["delaytime_cdf"]
            amplitude_cdf = self.uniform_to_pmt_ap[element]["amplitude_cdf"]

            delaytime_bin_size = self.uniform_to_pmt_ap[element]["delaytime_bin_size"]
            amplitude_bin_size = self.uniform_to_pmt_ap[element]["amplitude_bin_size"]

            # Assign each photon FRIST random uniform number rU0 from (0, 1] for timing
            rU0 = 1 - self.rng.random(len(merged_photon_timings))

            # delaytime_cdf is intentionally not normalized to 1 but the probability of the AP
            prob_ap = delaytime_cdf[merged_photon_channels, -1]
            if prob_ap.max() * self.pmt_ap_modifier > 0.5:
                prob = prob_ap.max() * self.pmt_ap_modifier
                self.log.warning(f"PMT after pulse probability is {prob} larger than 0.5?")

            # Scaling down (up) rU0 effectivly increase (decrease) the ap rate
            rU0 /= self.pmt_ap_modifier

            # Double the probability for those photon emitting dpe
            rU0[merged_photon_id_dpe] /= 2

            # Select those photons with U <= max of cdf of specific channel
            sel_photon_id = np.where(rU0 <= prob_ap)[0]
            if len(sel_photon_id) == 0:
                continue
            sel_photon_channel = merged_photon_channels[sel_photon_id]

            # Assign selected photon SECOND random uniform number rU1 from (0, 1] for amplitude
            rU1 = 1 - self.rng.random(len(sel_photon_channel))

            # The map is made so that the indices are delay time in unit of ns
            if "Uniform" in element:
                ap_delay = (
                    self.rng.uniform(
                        delaytime_cdf[sel_photon_channel, 0], delaytime_cdf[sel_photon_channel, 1]
                    )
                    * delaytime_bin_size
                )
                ap_amplitude = np.ones_like(ap_delay)
            else:
                ap_delay = (
                    np.argmin(
                        np.abs(delaytime_cdf[sel_photon_channel] - rU0[sel_photon_id][:, None]),
                        axis=-1,
                    )
                    * delaytime_bin_size
                    - self.pmt_ap_t_modifier
                )
                if len(amplitude_cdf.shape) == 2:
                    ap_amplitude = (
                        np.argmin(np.abs(amplitude_cdf[sel_photon_channel] - rU1[:, None]), axis=-1)
                        * amplitude_bin_size
                    )
                else:
                    ap_amplitude = (
                        np.argmin(np.abs(amplitude_cdf[None, :] - rU1[:, None]), axis=-1)
                        * amplitude_bin_size
                    )

            _photon_timings.append(merged_photon_timings[sel_photon_id] + ap_delay)
            _photon_channels.append(merged_photon_channels[sel_photon_id])
            _photon_amplitude.append(np.atleast_1d(ap_amplitude))

        if len(_photon_timings) > 0:
            _photon_timings = np.hstack(_photon_timings)
            _photon_channels = np.hstack(_photon_channels).astype(np.int64)
            _photon_amplitude = np.hstack(_photon_amplitude)
            _photon_gains = np.array(self.gains)[_photon_channels] * _photon_amplitude

            return _photon_timings, _photon_channels, _photon_gains

        else:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
