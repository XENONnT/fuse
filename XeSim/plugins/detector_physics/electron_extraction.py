import strax
import straxen
import numpy as np
from copy import deepcopy
import os
import logging

from ...common import make_map, make_patternmap

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('XeSim.detector_physics.electron_extraction')
log.setLevel('WARNING')


private_files_path = "path/to/private/files"
config = straxen.get_resource(os.path.join(private_files_path, 'sim_files/fax_config_nt_sr0_v4.json') , fmt='json')


@strax.takes_config(
    strax.Option('s2_correction_map_file',
                 default=os.path.join(private_files_path, "strax_files/XENONnT_s2_xy_map_v4_210503_mlp_3_in_1_iterated.json"),
                 track=False,
                 infer_type=False,
                 help="s2_correction_map"),
    strax.Option('s2_pattern_map_file',
                 default=os.path.join(private_files_path, "sim_files/XENONnT_s2_xy_patterns_GXe_LCE_corrected_qes_MCv4.3.0_wires.pkl"),
                 track=False,
                 infer_type=False,
                 help="s2_pattern_map"),
    strax.Option('to_pe_file', default=os.path.join(private_files_path, "sim_files/to_pe_nt.npy"), track=False, infer_type=False,
                 help="to_pe file"),
    strax.Option('digitizer_voltage_range', default=config['digitizer_voltage_range'], track=False, infer_type=False,
                 help="digitizer_voltage_range"),
    strax.Option('digitizer_bits', default=config['digitizer_bits'], track=False, infer_type=False,
                 help="digitizer_bits"),
    strax.Option('pmt_circuit_load_resistor', default=config['pmt_circuit_load_resistor'], track=False, infer_type=False,
                 help="pmt_circuit_load_resistor"),
    strax.Option('se_gain_from_map', default=config['se_gain_from_map'], track=False, infer_type=False,
                 help="se_gain_from_map"),
    strax.Option('se_gain_map',
                 default=os.path.join(private_files_path, "strax_files/XENONnT_se_xy_map_v1_mlp.json"),
                 track=False,
                 infer_type=False,
                 help="se_gain_map"),
    strax.Option('s2_secondary_sc_gain', default=config['s2_secondary_sc_gain'], track=False, infer_type=False,
                 help="s2_secondary_sc_gain"),
    strax.Option('g2_mean', default=config['g2_mean'], track=False, infer_type=False,
                 help="g2_mean"),
    strax.Option('electron_extraction_yield', default=config['electron_extraction_yield'], track=False, infer_type=False,
                 help="electron_extraction_yield"),
    strax.Option('ext_eff_from_map', default=config['ext_eff_from_map'], track=False, infer_type=False,
                 help="ext_eff_from_map"),
)
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
    
    def setup(self):

        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running ElectronExtraction in debug mode")
        
        to_pe = straxen.get_resource(self.to_pe_file, fmt='npy')
        self.to_pe = to_pe[0][1]
        
        adc_2_current = (self.digitizer_voltage_range
                / 2 ** (self.digitizer_bits)
                 / self.pmt_circuit_load_resistor)

        gains = np.divide(adc_2_current,
                          self.to_pe,
                          out=np.zeros_like(self.to_pe),
                          where=self.to_pe != 0)
        
        self.pmt_mask = np.array(gains) > 0  # Converted from to pe (from cmt by default)
        
        self.s2_pattern_map = make_patternmap(self.s2_pattern_map_file, fmt='pkl', pmt_mask=self.pmt_mask)
        
        if self.s2_correction_map_file:
            self.s2_correction_map = make_map(self.s2_correction_map_file, fmt = 'json')
        else:
            s2cmap = deepcopy(self.s2_pattern_map)
            # Lower the LCE by removing contribution from dead PMTs
            # AT: masking is a bit redundant due to PMT mask application in make_patternmap
            s2cmap.data['map'] = np.sum(s2cmap.data['map'][:][:], axis=2, keepdims=True, where=self.pmt_mask)
            # Scale by median value
            s2cmap.data['map'] = s2cmap.data['map'] / np.median(s2cmap.data['map'][s2cmap.data['map'] > 0])
            s2cmap.__init__(s2cmap.data)
            self.s2_correction_map = s2cmap
            
        self.se_gain_map = make_map(self.se_gain_map, fmt = "json")
    
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