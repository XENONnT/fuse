import strax
import straxen
import numpy as np
import logging

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.detector_physics.electron_extraction')
log.setLevel('WARNING')

@export
class ElectronExtraction(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ("microphysics_summary", "drifted_electrons")
    provides = "extracted_electrons"
    data_kind = "interactions_in_roi"
    
    #Forbid rechunking
    rechunk_on_save = False
    

    dtype = [('n_electron_extracted', np.int64),
            ]
    
    dtype = dtype + strax.time_fields

    #Config options
    debug = straxen.URLConfig(
        default=False, type=bool,track=False,
        help='Show debug informations',
    )

    digitizer_voltage_range = straxen.URLConfig(
        type=(int, float),
        help='digitizer_voltage_range',
    )

    digitizer_bits = straxen.URLConfig(
        type=(int, float),
        help='digitizer_bits',
    )

    pmt_circuit_load_resistor = straxen.URLConfig(
        type=(int, float),
        help='pmt_circuit_load_resistor',
    )

    s2_secondary_sc_gain = straxen.URLConfig(
        type=(int, float),
        help='s2_secondary_sc_gain',
    )
    #Rename? -> g2_value in beta_yields model 
    g2_mean = straxen.URLConfig(
        type=(int, float),
        help='g2_mean',
    )

    electron_extraction_yield = straxen.URLConfig(
        type=(int, float),
        help='electron_extraction_yield',
    )

    ext_eff_from_map = straxen.URLConfig(
        type=bool,
        help='ext_eff_from_map',
    )

    se_gain_from_map = straxen.URLConfig(
        type=bool,
        help='se_gain_from_map',
    )

    gains = straxen.URLConfig(
        cache=True,
        help='pmt gains',
    )
    
    s2_correction_map = straxen.URLConfig(
        cache=True,
        help='s2_correction_map',
    )
    
    se_gain_map = straxen.URLConfig(
        cache=True,
        help='se_gain_map',
    )
    
    s2_pattern_map = straxen.URLConfig(
        cache=True,
        help='s2_pattern_map',
    )

    
    def setup(self):

        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running ElectronExtraction in debug mode")
        
        self.pmt_mask = np.array(self.gains) > 0  # Converted from to pe (from cmt by default)
        
        #Is this else case ever used? if no -> remove
        #if self.s2_correction_map_file:
        #    self.s2_correction_map = make_map(self.s2_correction_map_file, fmt = 'json')
        #else:
        #    s2cmap = deepcopy(self.s2_pattern_map)
        #    # Lower the LCE by removing contribution from dead PMTs
        #    # AT: masking is a bit redundant due to PMT mask application in make_patternmap
        #    s2cmap.data['map'] = np.sum(s2cmap.data['map'][:][:], axis=2, keepdims=True, where=self.pmt_mask)
        #    # Scale by median value
        #    s2cmap.data['map'] = s2cmap.data['map'] / np.median(s2cmap.data['map'][s2cmap.data['map'] > 0])
        #    s2cmap.__init__(s2cmap.data)
        #    self.s2_correction_map = s2cmap
    
    def compute(self, interactions_in_roi):
        
        #Just apply this to clusters with photons
        mask = interactions_in_roi["electrons"] > 0

        if len(interactions_in_roi[mask]) == 0:
            return np.zeros(0, self.dtype)

        x = interactions_in_roi[mask]["x"]
        y = interactions_in_roi[mask]["y"]
        
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
                se_gains=rel_s2_cor*self.s2_secondary_sc_gain
            cy = self.g2_mean*rel_s2_cor/se_gains
        else:
            cy = self.electron_extraction_yield
            
        n_electron = np.random.binomial(n=interactions_in_roi[mask]["n_electron_interface"], p=cy)
        
        result = np.zeros(len(interactions_in_roi), dtype=self.dtype)
        result["n_electron_extracted"][mask] = n_electron
        result["time"] = interactions_in_roi["time"]
        result["endtime"] = interactions_in_roi["endtime"]
        
        return result