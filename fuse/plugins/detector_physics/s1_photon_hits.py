import strax
import straxen
import logging

import numpy as np
from copy import deepcopy

from ...common import FUSE_PLUGIN_TIMEOUT, pmt_gains

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.detector_physics.s1_photon_hits')

@export
class S1PhotonHits(strax.Plugin):

    __version__ = '0.1.2'

    depends_on = ("microphysics_summary")
    provides = "s1_photons"
    data_kind = "interactions_in_roi"

    #Forbid rechunking
    rechunk_on_save = False

    save_when = strax.SaveWhen.TARGET

    input_timeout = FUSE_PLUGIN_TIMEOUT

    dtype = [('n_s1_photon_hits', np.int32),
            ]
    dtype = dtype + strax.time_fields

    #Config options
    debug = straxen.URLConfig(
        default=False, type=bool,track=False,
        help='Show debug informations',
    )
    
    pmt_circuit_load_resistor = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=pmt_circuit_load_resistor",
        type=(int, float),
        cache=True,
        help='PMT circuit load resistor',
    )

    digitizer_bits = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=digitizer_bits",
        type=(int, float),
        cache=True,
        help='Number of bits of the digitizer boards',
    )

    digitizer_voltage_range = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=digitizer_voltage_range",
        type=(int, float),
        cache=True,
        help='Voltage range of the digitizer boards',
    )

    gain_model_mc = straxen.URLConfig(
        default="cmt://to_pe_model?version=ONLINE&run_id=plugin.run_id",
        infer_type=False,
        help='PMT gain model',
    )

    s1_pattern_map = straxen.URLConfig(
        default = 'pattern_map://resource://simulation_config://'
                  'SIMULATION_CONFIG_FILE.json?'
                  '&key=s1_pattern_map'
                  '&fmt=pkl'
                  '&pmt_mask=plugin.pmt_mask',
        cache=True,
        help='S1 pattern map',
    )

    p_double_pe_emision = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=p_double_pe_emision",
        type=(int, float),
        cache=True,
        help='Probability of double photo-electron emission',
    )

    s1_detection_efficiency = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=s1_detection_efficiency",
        type=(int, float),
        cache=True,
        help='S1 detection efficiency',
    )
    
    deterministic_seed = straxen.URLConfig(
        default=True, type=bool,
        help='Set the random seed from lineage and run_id, or pull the seed from the OS.',
    )

    def setup(self):
        
        if self.debug:
            log.setLevel('DEBUG')
            log.debug(f"Running S1PhotonHits version {self.__version__} in debug mode")
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

        self.gains = pmt_gains(self.gain_model_mc,
                               digitizer_voltage_range=self.digitizer_voltage_range,
                               digitizer_bits=self.digitizer_bits,
                               pmt_circuit_load_resistor=self.pmt_circuit_load_resistor
                               )

        self.pmt_mask = np.array(self.gains) > 0  # Converted from to pe (from cmt by default)

        #Build LCE map from s1 pattern map
        lcemap = deepcopy(self.s1_pattern_map)
        # AT: this scaling with mast is redundant to `make_patternmap`, but keep it in for now
        lcemap.data['map'] = np.sum(lcemap.data['map'][:][:][:], axis=3, keepdims=True, where=self.pmt_mask)
        lcemap.__init__(lcemap.data)
        self.s1_lce_correction_map = lcemap

    def compute(self, interactions_in_roi):

        #Just apply this to clusters with photons
        mask = interactions_in_roi["photons"] > 0

        if len(interactions_in_roi[mask]) == 0:
            return np.zeros(len(interactions_in_roi), self.dtype)
        
        x = interactions_in_roi[mask]['x']
        y = interactions_in_roi[mask]['y']
        z = interactions_in_roi[mask]['z']
        n_photons = interactions_in_roi[mask]['photons'].astype(np.int64)

        positions = np.array([x, y, z]).T

        n_photon_hits = self.get_n_photons(n_photons=n_photons,
                                           positions=positions,
                                           )
        
        result = np.zeros(interactions_in_roi.shape[0], dtype = self.dtype)

        result["time"] = interactions_in_roi["time"]
        result["endtime"] = interactions_in_roi["endtime"]

        result["n_s1_photon_hits"][mask] = n_photon_hits

        return result
    

    def get_n_photons(self, n_photons, positions):
    
        """Calculates number of detected photons based on number of photons in total and the positions
        :param n_photons: 1d array of ints with number of emitted S1 photons:
        :param positions: 2d array with xyz positions of interactions
        :param s1_lce_correction_map: interpolator instance of s1 light yield map
        :param config: dict wfsim config 

        return array with number photons"""
        ly = self.s1_lce_correction_map(positions)
        # depending on if you use the data driven or mc pattern map for light yield 
        #the shape of n_photon_hits will change. Mc needs a squeeze
        if len(ly.shape) != 1:
            ly = np.squeeze(ly, axis=-1)
        ly /= 1 + self.p_double_pe_emision
        ly *= self.s1_detection_efficiency

        n_photon_hits = self.rng.binomial(n=n_photons, p=ly)

        return n_photon_hits
