import strax
import numpy as np
import straxen
import logging

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.pmt_and_daq.pmt_afterpulses')

@export
class PMTAfterPulses(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ("propagated_s2_photons", "propagated_s1_photons")
    provides = "pmt_afterpulses"
    
    #Forbid rechunking
    rechunk_on_save = False

    save_when = strax.SaveWhen.TARGET
    
    data_kind = "AP_photons"
    
    dtype = [('channel', np.int64),
             ('dpe', np.bool_),
             ('photon_gain', np.int64),
            ]
    dtype = dtype + strax.time_fields

    #Config options
    debug = straxen.URLConfig(
        default=False, type=bool,track=False,
        help='Show debug informations',
    )

    pmt_ap_t_modifier = straxen.URLConfig(
        type=(int, float),
        help='pmt_ap_t_modifier',
    )

    pmt_ap_modifier = straxen.URLConfig(
        type=(int, float),
        help='pmt_ap_modifier',
    )

    pmt_circuit_load_resistor = straxen.URLConfig(
        type=(int, float),
        help='pmt_circuit_load_resistor',
    )

    digitizer_bits = straxen.URLConfig(
        type=(int, float),
        help='digitizer_bits',
    )

    digitizer_voltage_range = straxen.URLConfig(
        type=(int, float),
        help='digitizer_voltage_range',
    )
    
    gains = straxen.URLConfig(
        cache=True,
        help='pmt gains',
    )
    
    photon_ap_cdfs = straxen.URLConfig(
        cache=True,
        help='photon_ap_cdfs',
    )

    deterministic_seed = straxen.URLConfig(
        default=True, type=bool,
        help='Set the random seed from lineage and run_id, or pull the seed from the OS.',
    )
 
    def setup(self):

        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running PMTAfterPulses in debug mode")
        else: 
            log.setLevel('WARNING')

        if self.deterministic_seed:
            hash_string = strax.deterministic_hash((self.run_id, self.lineage))
            seed = int(hash_string.encode().hex(), 16)
            self.rng = np.random.default_rng(seed = seed)
            log.debug(f"Generating random numbers from seed {seed}")
        else: 
            self.rng = np.random.default_rng()
            log.debug(f"Generating random numbers with seed pulled from OS")
        
        self.uniform_to_pmt_ap = self.photon_ap_cdfs

        for k in self.uniform_to_pmt_ap.keys():
            for q in self.uniform_to_pmt_ap[k].keys():
                if isinstance(self.uniform_to_pmt_ap[k][q], list):
                    self.uniform_to_pmt_ap[k][q] = np.array(self.uniform_to_pmt_ap[k][q])
    
    def compute(self, S1_photons, S2_photons):
        
        if len(S1_photons) == 0 and len(S2_photons) == 0:
            return np.zeros(0, dtype=self.dtype)

        merged_photons = np.concatenate([S1_photons, S2_photons])
        S1_photons = None
        S2_photons = None
        
        #Sort all photons by time
        sortind = np.argsort(merged_photons["time"])
        merged_photons = merged_photons[sortind]
        
        _photon_timings = merged_photons["time"]
        _photon_channels = merged_photons["channel"]
        _photon_is_dpe = merged_photons["dpe"]

        ap_photon_timings, ap_photon_channels, ap_photon_gains = self.photon_afterpulse(_photon_timings,_photon_channels,_photon_is_dpe)
        ap_photon_is_dpe = np.zeros_like(ap_photon_timings).astype(np.bool_)

        result = np.zeros(len(ap_photon_channels), dtype = self.dtype)
        result["channel"] = ap_photon_channels
        result["time"] = ap_photon_timings
        result["endtime"] = ap_photon_timings
        result["dpe"] = ap_photon_is_dpe
        result["photon_gain"] = ap_photon_gains
        
        return result

    
    def photon_afterpulse(self, merged_photon_timings, merged_photon_channels, merged_photon_id_dpe):
        """
        For pmt afterpulses, gain and dpe generation is a bit different from standard photons
        """
        element_list = self.uniform_to_pmt_ap.keys()
        _photon_timings = []
        _photon_channels = []
        _photon_amplitude = []

        for element in element_list:
            delaytime_cdf = self.uniform_to_pmt_ap[element]['delaytime_cdf']
            amplitude_cdf = self.uniform_to_pmt_ap[element]['amplitude_cdf']

            delaytime_bin_size = self.uniform_to_pmt_ap[element]['delaytime_bin_size']
            amplitude_bin_size = self.uniform_to_pmt_ap[element]['amplitude_bin_size']

            # Assign each photon FRIST random uniform number rU0 from (0, 1] for timing
            rU0 = 1 - self.rng.random(len(merged_photon_timings))

            # delaytime_cdf is intentionally not normalized to 1 but the probability of the AP 
            prob_ap = delaytime_cdf[merged_photon_channels, -1]
            if prob_ap.max() * self.pmt_ap_modifier > 0.5:
                prob = prob_ap.max() * self.pmt_ap_modifier
                log.warning(f'PMT after pulse probability is {prob} larger than 0.5?')

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
            if 'Uniform' in element:
                ap_delay = (self.rng.uniform(delaytime_cdf[sel_photon_channel, 0], 
                                            delaytime_cdf[sel_photon_channel, 1])
                            * delaytime_bin_size)
                ap_amplitude = np.ones_like(ap_delay)
            else:
                ap_delay = (np.argmin(
                    np.abs(
                        delaytime_cdf[sel_photon_channel]
                        - rU0[sel_photon_id][:, None]), axis=-1) * delaytime_bin_size
                            - self.pmt_ap_t_modifier)
                if len(amplitude_cdf.shape) == 2:
                    ap_amplitude = np.argmin(
                        np.abs(
                            amplitude_cdf[sel_photon_channel]
                            - rU1[:, None]), axis=-1) * amplitude_bin_size
                else:
                    ap_amplitude = np.argmin(
                        np.abs(
                            amplitude_cdf[None, :]
                            - rU1[:, None]), axis=-1) * amplitude_bin_size

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