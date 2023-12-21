import strax
import straxen
import numpy as np
import os
import logging

from ...common import pmt_gains, FUSE_PLUGIN_TIMEOUT

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.detector_physics.electron_extraction')

@export
class ElectronExtraction(strax.Plugin):
    
    __version__ = "0.1.1"
    
    depends_on = ("microphysics_summary", "drifted_electrons")
    provides = "extracted_electrons"
    data_kind = "interactions_in_roi"
    
    #Forbid rechunking
    rechunk_on_save = False
    
    save_when = strax.SaveWhen.TARGET

    input_timeout = FUSE_PLUGIN_TIMEOUT

    dtype = [('n_electron_extracted', np.int32),
            ]
    
    dtype = dtype + strax.time_fields

    #Config options
    debug = straxen.URLConfig(
        default=False, type=bool,track=False,
        help='Show debug informations',
    )

    digitizer_voltage_range = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=digitizer_voltage_range",
        type=(int, float),
        help='Voltage range of the digitizer boards',
    )

    digitizer_bits = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=digitizer_bits",
        type=(int, float),
        help='Number of bits of the digitizer boards',
    )

    pmt_circuit_load_resistor = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=pmt_circuit_load_resistor",
        type=(int, float),
        help='PMT circuit load resistor ',
    )

    s2_secondary_sc_gain_mc = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=s2_secondary_sc_gain",
        type=(int, float),
        help='Secondary scintillation gain',
    )
    #Rename? -> g2_value in beta_yields model 
    g2_mean = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=g2_mean",
        type=(int, float),
        help='mean value of the g2 gain. ',
    )

    electron_extraction_yield = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=electron_extraction_yield",
        type=(int, float),
        help='Electron extraction yield',
    )

    ext_eff_from_map = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=ext_eff_from_map",
        type=bool,
        help='Boolean indication if the extraction efficiency is taken from a map',
    )

    se_gain_from_map = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=se_gain_from_map",
        type=bool,
        help='Boolean indication if the secondary scintillation gain is taken from a map',
    )

    gain_model_mc = straxen.URLConfig(
        default="cmt://to_pe_model?version=ONLINE&run_id=plugin.run_id",
        infer_type=False,
        help='PMT gain model',
    )
    
    s2_correction_map = straxen.URLConfig(
        default = 'itp_map://resource://simulation_config://'
                  'SIMULATION_CONFIG_FILE.json?'
                  '&key=s2_correction_map'
                  '&fmt=json',
        cache=True,
        help='S2 correction map',
    )
    
    se_gain_map = straxen.URLConfig(
        default = 'itp_map://resource://simulation_config://'
                  'SIMULATION_CONFIG_FILE.json?'
                  '&key=se_gain_map'
                  '&fmt=json',
        cache=True,
        help='Map of the single electron gain',
    )
    
    s2_pattern_map = straxen.URLConfig(
        default = 'pattern_map://resource://simulation_config://'
                  'SIMULATION_CONFIG_FILE.json?'
                  '&key=s2_pattern_map'
                  '&fmt=pkl'
                  '&pmt_mask=plugin.pmt_mask',
        cache=True,
        help='S2 pattern map',
    )

    deterministic_seed = straxen.URLConfig(
        default=True, type=bool,
        help='Set the random seed from lineage and run_id, or pull the seed from the OS.',
    )
    
    def setup(self):

        if self.debug:
            log.setLevel('DEBUG')
            log.debug(f"Running ElectronExtraction version {self.__version__} in debug mode")
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

        self.gains = self.gains = pmt_gains(self.gain_model_mc,
                               digitizer_voltage_range=self.digitizer_voltage_range,
                               digitizer_bits=self.digitizer_bits,
                               pmt_circuit_load_resistor=self.pmt_circuit_load_resistor
                               )

        self.pmt_mask = np.array(self.gains) > 0 
    def compute(self, interactions_in_roi):
        
        #Just apply this to clusters with photons
        mask = interactions_in_roi["electrons"] > 0

        if len(interactions_in_roi[mask]) == 0:
            return np.zeros(0, self.dtype)

        x = interactions_in_roi[mask]["x_obs"]
        y = interactions_in_roi[mask]["y_obs"]
        
        xy_int = np.array([x, y]).T # maps are in R_true, so orginal position should be here

        if self.ext_eff_from_map:
            # Extraction efficiency is g2(x,y)/SE_gain(x,y)
            rel_s2_cor=self.s2_correction_map(xy_int)
            #doesn't always need to be flattened, but if s2_correction_map = False, then map is made from MC
            rel_s2_cor = rel_s2_cor.flatten()

            if self.se_gain_from_map:
                se_gains=self.se_gain_map(xy_int)
            else:
                # is in get_s2_light_yield map is scaled according to relative s2 correction
                # we also need to do it here to have consistent g2
                se_gains=rel_s2_cor*self.s2_secondary_sc_gain_mc
            cy = self.g2_mean*rel_s2_cor/se_gains
        else:
            cy = self.electron_extraction_yield
            
        n_electron = self.rng.binomial(n=interactions_in_roi[mask]["n_electron_interface"], p=cy)
        
        result = np.zeros(len(interactions_in_roi), dtype=self.dtype)
        result["n_electron_extracted"][mask] = n_electron
        result["time"] = interactions_in_roi["time"]
        result["endtime"] = interactions_in_roi["endtime"]
        
        return result