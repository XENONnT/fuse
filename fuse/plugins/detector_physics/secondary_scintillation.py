import strax
import numpy as np
import straxen
import logging
from immutabledict import immutabledict

from ...common import pmt_gains, FUSE_PLUGIN_TIMEOUT

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.detector_physics.secondary_scintillation')

@export
class SecondaryScintillation(strax.Plugin):
    
    __version__ = "0.1.4"

    result_name_photons = "s2_photons"
    result_name_photons_sum = "s2_photons_sum"
    
    depends_on = ("drifted_electrons","extracted_electrons" ,"electron_time", "microphysics_summary")
    provides = (result_name_photons, result_name_photons_sum)
    data_kind = {result_name_photons: "individual_electrons",
                 result_name_photons_sum : "interactions_in_roi"
                }
    
    dtype_photons = [('n_s2_photons', np.int32),] + strax.time_fields
    dtype_sum_photons = [('sum_s2_photons', np.int32),] + strax.time_fields
    
    dtype = dict()
    dtype[result_name_photons] = dtype_photons
    dtype[result_name_photons_sum] = dtype_sum_photons

    #Forbid rechunking
    rechunk_on_save = False

    save_when = immutabledict({result_name_photons:strax.SaveWhen.TARGET,
                               result_name_photons_sum:strax.SaveWhen.ALWAYS
                              })

    input_timeout = FUSE_PLUGIN_TIMEOUT
    
    #Config options
    debug = straxen.URLConfig(
        default=False, type=bool,track=False,
        help='Show debug informations',
    )

    #Move this into the config
    s2_gain_spread = straxen.URLConfig(
        default = 0,
        type=(int, float),
        help='Spread of the S2 gain',
    )

    s2_secondary_sc_gain_mc = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=s2_secondary_sc_gain",
        type=(int, float),
        cache=True,
        help='Secondary scintillation gain',
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

    se_gain_from_map = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=se_gain_from_map",
        cache=True,
        help='Boolean indication if the secondary scintillation gain is taken from a map',
    )

    p_double_pe_emision = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=p_double_pe_emision",
        type=(int, float),
        cache=True,
        help='Probability of double photo-electron emission',
    )
    
    se_gain_map = straxen.URLConfig(
        default = 'itp_map://resource://simulation_config://'
                  'SIMULATION_CONFIG_FILE.json?'
                  '&key=se_gain_map'
                  '&fmt=json',
        cache=True,
        help='Map of the single electron gain ',
    )
    
    s2_correction_map = straxen.URLConfig(
        default = 'itp_map://resource://simulation_config://'
                  'SIMULATION_CONFIG_FILE.json?'
                  '&key=s2_correction_map'
                  '&fmt=json',
        cache=True,
        help='S2 correction map',
    )
    
    gain_model_mc = straxen.URLConfig(
        default="cmt://to_pe_model?version=ONLINE&run_id=plugin.run_id",
        infer_type=False,
        help='PMT gain model',
    )

    n_top_pmts = straxen.URLConfig(
        type=(int),
        help='Number of PMTs on top array',
    )

    n_tpc_pmts = straxen.URLConfig(
        type=(int),
        help='Number of PMTs in the TPC',
    )

    s2_mean_area_fraction_top = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=s2_mean_area_fraction_top",
        type=(int, float),
        cache=True,
        help='Mean S2 area fraction top',
    )
    
    s2_pattern_map = straxen.URLConfig(
        default = 's2_aft_scaling://pattern_map://resource://simulation_config://'
                  'SIMULATION_CONFIG_FILE.json?'
                  '&key=s2_pattern_map'
                  '&fmt=pkl'
                  '&pmt_mask=plugin.pmt_mask'
                  '&s2_mean_area_fraction_top=plugin.s2_mean_area_fraction_top'
                  '&n_tpc_pmts=plugin.n_tpc_pmts'
                  '&n_top_pmts=plugin.n_top_pmts'
                  ,
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
            log.debug(f"Running SecondaryScintillation version {self.__version__} in debug mode")
        else: 
            log.setLevel('INFO')

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

        self.pmt_mask = np.array(self.gains)
    
    def compute(self, interactions_in_roi, individual_electrons):
        
        #Just apply this to clusters with electrons
        mask = interactions_in_roi["n_electron_extracted"] > 0

        if len(interactions_in_roi[mask]) == 0:
            empty_result = np.zeros(len(interactions_in_roi), self.dtype[self.result_name_photons_sum])
            empty_result["time"] = interactions_in_roi["time"]
            empty_result["endtime"] = interactions_in_roi["endtime"]
            
            return {self.result_name_photons : np.zeros(0, self.dtype[self.result_name_photons]),
                    self.result_name_photons_sum : empty_result}
        
        positions = np.array([individual_electrons["x"], individual_electrons["y"]]).T
        
        electron_gains = self.get_s2_light_yield(positions=positions)
        
        n_photons_per_ele = self.rng.poisson(electron_gains)
        
        if self.s2_gain_spread:
            n_photons_per_ele += self.rng.normal(0, self.s2_gain_spread, len(n_photons_per_ele)).astype(np.int64)
        n_photons_per_ele[n_photons_per_ele < 0] = 0
        
        result_photons = np.zeros(len(n_photons_per_ele), dtype = self.dtype[self.result_name_photons])
        result_photons["n_s2_photons"] = n_photons_per_ele
        result_photons["time"] = individual_electrons["time"]
        result_photons["endtime"] = individual_electrons["endtime"]
        
        #Calculate the sum of photons per interaction
        grouped_result_photons, unique_cluster_id = group_result_photons_by_cluster_id(result_photons, individual_electrons["cluster_id"])
        sum_photons_per_interaction = np.array([np.sum(element["n_s2_photons"]) for element in grouped_result_photons])
        
        #Bring sum_photons_per_interaction into the same cluster order as interactions_in_roi
        #Maybe this line is too complicated...
        sum_photons_per_interaction_reordered = [sum_photons_per_interaction[np.argwhere(unique_cluster_id == element)[0][0]] for element in interactions_in_roi["cluster_id"][mask]]
        
        result_sum_photons = np.zeros(len(interactions_in_roi), dtype = self.dtype[self.result_name_photons_sum])
        result_sum_photons["sum_s2_photons"][mask] = sum_photons_per_interaction
        result_sum_photons["time"] = interactions_in_roi["time"]
        result_sum_photons["endtime"]= interactions_in_roi["endtime"]

        return {self.result_name_photons : strax.sort_by_time(result_photons),
                self.result_name_photons_sum : result_sum_photons}
        
        
    def get_s2_light_yield(self, positions):
        """Calculate s2 light yield...

        :param positions: 2d array of positions (floats)

        returns array of floats (mean expectation) 
        """

        if self.se_gain_from_map:
            sc_gain = self.se_gain_map(positions)
        else:
            # calculate it from MC pattern map directly if no "se_gain_map" is given
            sc_gain = self.s2_correction_map(positions)
            sc_gain *= self.s2_secondary_sc_gain_mc

        # depending on if you use the data driven or mc pattern map for light yield for S2
        # the shape of n_photon_hits will change. Mc needs a squeeze
        if len(sc_gain.shape) != 1:
            sc_gain=np.squeeze(sc_gain, axis=-1)

        # sc gain should has the unit of pe / electron, here we divide 1 + dpe to get nphoton / electron
        sc_gain /= 1 + self.p_double_pe_emision

        # data driven map contains nan, will be set to 0 here
        sc_gain[np.isnan(sc_gain)] = 0
        
        return sc_gain

def group_result_photons_by_cluster_id(result, cluster_id):
    """Function to group result_photons by cluster_id"""
    
    sort_index = np.argsort(cluster_id)
    
    cluster_id_sorted = cluster_id[sort_index]
    result_sorted = result[sort_index]

    unique_cluster_id, split_position = np.unique(cluster_id_sorted, return_index=True)
    return np.split(result, split_position[1:]), unique_cluster_id