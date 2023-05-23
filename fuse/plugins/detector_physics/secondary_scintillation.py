import strax
import numpy as np
import straxen
import logging

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.detector_physics.secondary_scintillation')
log.setLevel('WARNING')

@export
class SecondaryScintillation(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ("drifted_electrons","extracted_electrons" ,"electron_time")
    provides = ("s2_photons", "s2_photons_sum")
    data_kind = {"s2_photons": "individual_electrons",
                 "s2_photons_sum" : "interactions_in_roi"
                }
    
    dtype_photons = [('n_s2_photons', np.int64),] + strax.time_fields
    dtype_sum_photons = [('sum_s2_photons', np.int64),] + strax.time_fields
    
    dtype = dict()
    dtype["s2_photons"] = dtype_photons
    dtype["s2_photons_sum"] = dtype_sum_photons

    #Forbid rechunking
    rechunk_on_save = False
    
    #Config options
    debug = straxen.URLConfig(
        default=False, type=bool,track=False,
        help='Show debug informations',
    )

    s2_gain_spread = straxen.URLConfig(
        type=(int, float),
        help='s2_gain_spread',
    )

    s2_secondary_sc_gain = straxen.URLConfig(
        type=(int, float),
        help='s2_secondary_sc_gain',
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

    se_gain_from_map = straxen.URLConfig(
        help='se_gain_from_map',
    )

    p_double_pe_emision = straxen.URLConfig(
        type=(int, float),
        help='p_double_pe_emision',
    )
    
    se_gain_map = straxen.URLConfig(
        cache=True,
        help='se_gain_map',
    )
    
    s2_correction_map = straxen.URLConfig(
        cache=True,
        help='s2_correction_map',
    )
    
    gains = straxen.URLConfig(
        cache=True,
        help='pmt gains',
    )
    
    s2_pattern_map = straxen.URLConfig(
        cache=True,
        help='s2_pattern_map',
    )

    fixed_seed = straxen.URLConfig(
        default=True, type=bool,
        help='Set the random seed from lineage and run_id, or pull the seed from the OS.',
    )

    def setup(self):
        
        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running SecondaryScintillation in debug mode")

        if self.fixed_seed:
            hash_string = strax.deterministic_hash((self.run_id, self.lineage))
            seed = int(hash_string.encode().hex(), 16)
            self.rng = np.random.default_rng(seed = seed)
            log.debug(f"Generating random numbers from seed {seed}")
        else: 
            self.rng = np.random.default_rng()
            log.debug(f"Generating random numbers with seed pulled from OS")
        
        self.pmt_mask = np.array(self.gains)
        
        #Are these if cases needed?? -> If no remove, if yes, correct the code
        #if self.s2_correction_map_file:
        #    self.s2_correction_map = make_map(self.s2_correction_map_file, fmt = 'json')
        #else:

        #    self.pmt_mask = np.array(self.gains) > 0  # Converted from to pe (from cmt by default)

        #    self.s2_pattern_map = make_patternmap(self.s2_pattern_map_file, fmt='pkl', pmt_mask=self.pmt_mask)
            
            
        #    s2cmap = deepcopy(self.s2_pattern_map)
            # Lower the LCE by removing contribution from dead PMTs
            # AT: masking is a bit redundant due to PMT mask application in make_patternmap
        #    s2cmap.data['map'] = np.sum(s2cmap.data['map'][:][:], axis=2, keepdims=True, where=self.pmt_mask)
            # Scale by median value
        #    s2cmap.data['map'] = s2cmap.data['map'] / np.median(s2cmap.data['map'][s2cmap.data['map'] > 0])
        #    s2cmap.__init__(s2cmap.data)
        #    self.s2_correction_map = s2cmap
    
    def compute(self, interactions_in_roi, individual_electrons):
        
        #Just apply this to clusters with photons
        mask = interactions_in_roi["n_electron_extracted"] > 0

        if len(interactions_in_roi[mask]) == 0:
            return dict(s2_photons=np.zeros(0, self.dtype["s2_photons"]),
                        s2_photons_sum=np.zeros(0, self.dtype["s2_photons_sum"]))
        
        positions = np.array([interactions_in_roi[mask]["x"], interactions_in_roi[mask]["y"]]).T
        
        sc_gain = self.get_s2_light_yield(positions=positions)
        
        electron_gains = np.repeat(sc_gain, interactions_in_roi[mask]["n_electron_extracted"])
        
        n_photons_per_ele = self.rng.poisson(electron_gains)
        
        if self.s2_gain_spread:
            n_photons_per_ele += self.rng.normal(0, self.s2_gain_spread, len(n_photons_per_ele)).astype(np.int64)
        
        sum_photons_per_interaction = [np.sum(x) for x in np.split(n_photons_per_ele, np.cumsum(interactions_in_roi[mask]["n_electron_extracted"]))[:-1]]
        
        n_photons_per_ele[n_photons_per_ele < 0] = 0
        
        result_photons = np.zeros(len(n_photons_per_ele), dtype = self.dtype["s2_photons"])
        result_photons["n_s2_photons"] = n_photons_per_ele
        result_photons["time"] = individual_electrons["time"]
        result_photons["endtime"] = individual_electrons["endtime"]
        
        result_sum_photons = np.zeros(len(interactions_in_roi), dtype = self.dtype["s2_photons_sum"])
        result_sum_photons["sum_s2_photons"][mask] = sum_photons_per_interaction
        result_sum_photons["time"][mask] = interactions_in_roi[mask]["time"]
        result_sum_photons["endtime"][mask] = interactions_in_roi[mask]["endtime"]
        
        return dict(s2_photons=result_photons,
                    s2_photons_sum=result_sum_photons)
        
        
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
            sc_gain *= self.s2_secondary_sc_gain

        # depending on if you use the data driven or mc pattern map for light yield for S2
        # the shape of n_photon_hits will change. Mc needs a squeeze
        if len(sc_gain.shape) != 1:
            sc_gain=np.squeeze(sc_gain, axis=-1)

        # sc gain should has the unit of pe / electron, here we divide 1 + dpe to get nphoton / electron
        sc_gain /= 1 + self.p_double_pe_emision

        # data driven map contains nan, will be set to 0 here
        sc_gain[np.isnan(sc_gain)] = 0
        
        return sc_gain