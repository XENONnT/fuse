import strax
import straxen
import numpy as np
import os
import logging

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.detector_physics.electron_extraction')
log.setLevel('WARNING')

base_path = os.path.abspath(os.getcwd())
private_files_path = os.path.join("/",*base_path.split("/")[:-2], "private_nt_aux_files")
config = straxen.get_resource(os.path.join(private_files_path, 'sim_files/fax_config_nt_sr0_v4.json') , fmt='json')

@export
class ElectronExtraction(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ("microphysics_summary", "drifted_electrons")
    provides = "extracted_electrons"
    data_kind = "electron_cloud"
    
    #Forbid rechunking
    rechunk_on_save = False
    

    dtype = [('n_electron_extracted', np.int64),
            ]
    
    dtype = dtype + strax.time_fields

    #Config options
    debug = straxen.URLConfig(
        default=False, type=bool,
        help='Show debug informations',
    )

    digitizer_voltage_range = straxen.URLConfig(
        default=config["digitizer_voltage_range"], type=(int, float),
        help='digitizer_voltage_range',
    )

    digitizer_bits = straxen.URLConfig(
        default=config["digitizer_bits"], type=(int, float),
        help='digitizer_bits',
    )

    pmt_circuit_load_resistor = straxen.URLConfig(
        default=config["pmt_circuit_load_resistor"], type=(int, float),
        help='pmt_circuit_load_resistor',
    )

    s2_secondary_sc_gain = straxen.URLConfig(
        default=config["s2_secondary_sc_gain"], type=(int, float),
        help='s2_secondary_sc_gain',
    )
    #Rename? -> g2_value in beta_yields model 
    g2_mean = straxen.URLConfig(
        default=config["g2_mean"], type=(int, float),
        help='g2_mean',
    )

    electron_extraction_yield = straxen.URLConfig(
        default=config["electron_extraction_yield"], type=(int, float),
        help='electron_extraction_yield',
    )

    ext_eff_from_map = straxen.URLConfig(
        default=config['ext_eff_from_map'], type=bool,
        help='ext_eff_from_map',
    )

    se_gain_from_map = straxen.URLConfig(
        default=config['se_gain_from_map'], type=bool,
        help='se_gain_from_map',
    )

    gains = straxen.URLConfig(
        default='pmt_gains://resource://format://'
                f'{os.path.join(private_files_path,"sim_files/to_pe_nt.npy")}?'
                '&fmt=npy'
                f'&digitizer_voltage_range=plugin.digitizer_voltage_range'
                f'&digitizer_bits=plugin.digitizer_bits'
                f'&pmt_circuit_load_resistor=plugin.pmt_circuit_load_resistor',
        cache=True,
        help='pmt gains',
    )
    
    s2_correction_map = straxen.URLConfig(
        default='itp_map://resource://format://'
                f'{os.path.join(private_files_path, "strax_files/XENONnT_s2_xy_map_v4_210503_mlp_3_in_1_iterated.json")}?'
                '&fmt=json',
        cache=True,
        help='s2_correction_map',
    )
    
    se_gain_map = straxen.URLConfig(
        default='itp_map://resource://format://'
                f'{os.path.join(private_files_path, "strax_files/XENONnT_se_xy_map_v1_mlp.json")}?'
                '&fmt=json',
        cache=True,
        help='se_gain_map',
    )
    
    s2_pattern_map = straxen.URLConfig(
        default='pattern_map://resource://format://'
                f'{os.path.join(private_files_path, "sim_files/XENONnT_s2_xy_patterns_GXe_LCE_corrected_qes_MCv4.3.0_wires.pkl")}?'
                '&fmt=pkl'
                '&pmt_mask=plugin.pmt_mask',
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
    
    def compute(self, clustered_interactions, electron_cloud):
        
        #Just apply this to clusters with free electrons
        instruction = clustered_interactions[clustered_interactions["electrons"] > 0]

        if len(instruction) == 0:
            return np.zeros(0, self.dtype)

        x = instruction["x"]
        y = instruction["y"]
        
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
            
        n_electron = np.random.binomial(n=electron_cloud["n_electron_interface"], p=cy)
        
        result = np.zeros(len(n_electron), dtype=self.dtype)
        result["n_electron_extracted"] = n_electron
        result["time"] = instruction["time"]
        result["endtime"] = instruction["endtime"]
        
        return result