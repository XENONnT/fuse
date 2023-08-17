# Define simulation cotexts here
# When fuse gets public this needs to be moved to e.g. cutax?

import strax
import cutax
import straxen
import fuse
import os
import numpy as np

from straxen import URLConfig

#Microphysics context
def microphysics_context(out_dir):
    st = strax.Context(register = [fuse.micro_physics.ChunkInput,
                                   fuse.micro_physics.FindCluster,
                                   fuse.micro_physics.MergeCluster,
                                   fuse.micro_physics.XENONnT_TPC,
                                   fuse.micro_physics.XENONnT_BelowCathode,
                                   fuse.micro_physics.VolumesMerger,
                                   fuse.micro_physics.ElectricField,
                                   fuse.micro_physics.NestYields,
                                   fuse.micro_physics.MicroPhysicsSummary],
                    storage = [strax.DataDirectory(out_dir)]
                    )

    st.set_config({
        "efield_map": 'itp_map://resource://format://'
                      'fieldmap_2D_B2d75n_C2d75n_G0d3p_A4d9p_T0d9n_PMTs1d3n_FSR0d65p_QPTFE_0d5n_0d4p.json.gz?'
                      '&fmt=json.gz'
                      '&method=RegularGridInterpolator',
        #'g1_value' : 0.151, 
        #'g2_value' : 16.45,
        #'cs1_spline_path': '/project2/lgrandi/pkavrigin/2023-04-24_epix_data_files/cs1_func_E_option2.pkl',
        #'cs2_spline_path' : '/project2/lgrandi/pkavrigin/2023-04-24_epix_data_files/cs2_func_E_option2.pkl',
    })
    
    return st


#Full Chain Context
#For now lets just hijack the cutax simulations context 

full_chain_modules = [fuse.micro_physics.ChunkInput,
                      fuse.micro_physics.FindCluster,
                      fuse.micro_physics.MergeCluster,
                      fuse.micro_physics.XENONnT_TPC,
                      fuse.micro_physics.XENONnT_BelowCathode,
                      fuse.micro_physics.VolumesMerger,
                      fuse.micro_physics.ElectricField,
                      fuse.micro_physics.NestYields,
                      fuse.micro_physics.MicroPhysicsSummary,
                      fuse.detector_physics.S1PhotonHits,
                      fuse.detector_physics.S1PhotonPropagation,
                      fuse.detector_physics.ElectronDrift,
                      fuse.detector_physics.ElectronExtraction,
                      fuse.detector_physics.ElectronTiming,
                      fuse.detector_physics.SecondaryScintillation,
                      fuse.detector_physics.S2PhotonPropagation,
                      fuse.pmt_and_daq.PMTAfterPulses,
                      fuse.pmt_and_daq.PhotonSummary,
                      fuse.pmt_and_daq.PMTResponseAndDAQ,
                     ]

def full_chain_context(out_dir, config):

    st = cutax.contexts.xenonnt_sim_SR0v3_cmt_v9(output_folder = out_dir)

    for module in full_chain_modules:
        st.register(module)

    st.set_config({
        "detector": "XENONnT",
        "efield_map": 'itp_map://resource://format://'
                      'fieldmap_2D_B2d75n_C2d75n_G0d3p_A4d9p_T0d9n_PMTs1d3n_FSR0d65p_QPTFE_0d5n_0d4p.json.gz?'
                      '&fmt=json.gz'
                      '&method=RegularGridInterpolator',
        "drift_velocity_liquid": config["drift_velocity_liquid"],
        "drift_time_gate": config["drift_time_gate"],
        "diffusion_constant_longitudinal": config["diffusion_constant_longitudinal"],
        "electron_lifetime_liquid": config["electron_lifetime_liquid"],
        "enable_field_dependencies": config["enable_field_dependencies"],
        "tpc_length": config["tpc_length"],
        "field_distortion_model": config["field_distortion_model"],
        "field_dependencies_map_tmp": 'itp_map://resource://format://'
                                      'field_dependent_radius_depth_maps_B2d75n_C2d75n_G0d3p_A4d9p_T0d9n_PMTs1d3n_FSR0d65p_QPTFE_0d5n_0d4p.json.gz?'
                                      '&fmt=json.gz'
                                      '&method=RectBivariateSpline',
        "diffusion_longitudinal_map_tmp":'itp_map://resource://format://'
                                         'data_driven_diffusion_map_XENONnTSR0V2.json.gz?'
                                         '&fmt=json.gz'
                                         '&method=WeightedNearestNeighbors',
        "fdc_map_fuse": 'itp_map://resource://format://'
                        'init_to_final_position_mapping_B2d75n_C2d75n_G0d3p_A4d9p_T0d9n_PMTs1d3n_FSR0d65p_QPTFE_0d5n_0d4p.json.gz?'
                        '&fmt=json.gz'
                        '&method=RectBivariateSpline',
        "digitizer_voltage_range": config["digitizer_voltage_range"],
        "digitizer_bits": config["digitizer_bits"],
        "pmt_circuit_load_resistor": config["pmt_circuit_load_resistor"],
        "s2_secondary_sc_gain": config["s2_secondary_sc_gain"],
        "g2_mean": config["g2_mean"],
        "electron_extraction_yield": config["electron_extraction_yield"],
        "ext_eff_from_map": config["ext_eff_from_map"],
        "se_gain_from_map": config["se_gain_from_map"],
        "gains": 'pmt_gains://resource://format://'
                 'to_pe_nt.npy?'
                 '&fmt=npy'
                 '&digitizer_voltage_range=plugin.digitizer_voltage_range'
                 '&digitizer_bits=plugin.digitizer_bits'
                 '&pmt_circuit_load_resistor=plugin.pmt_circuit_load_resistor',
        "s2_correction_map": 'itp_map://resource://format://'
                             'XENONnT_s2_xy_map_v4_210503_mlp_3_in_1_iterated.json?'
                             '&fmt=json',
        "se_gain_map": 'itp_map://resource://format://'
                       'XENONnT_se_xy_map_v1_mlp.json?'
                       '&fmt=json',
        "s2_pattern_map": 'pattern_map://resource://format://'
                          'XENONnT_s2_xy_patterns_GXe_LCE_corrected_qes_MCv4.3.0_wires.pkl?'
                          '&fmt=pkl'
                          '&pmt_mask=plugin.pmt_mask',
        "electron_trapping_time": config["electron_trapping_time"],
        "p_double_pe_emision": config["p_double_pe_emision"],
        "pmt_transit_time_spread": config["pmt_transit_time_spread"],
        "pmt_transit_time_mean": config["pmt_transit_time_mean"],
        "maximum_recombination_time": config["maximum_recombination_time"],
        "n_top_pmts": 253,
        "n_tpc_pmts": 494,
        "s1_detection_efficiency": 1,
        "s1_lce_correction_map": 'itp_map://resource://format://'
                                 'XENONnT_s1_xyz_LCE_corrected_qes_MCva43fa9b_wires.json.gz?'
                                 '&fmt=json.gz',
        "s1_pattern_map":'pattern_map://resource://format://'
                        'XENONnT_s1_xyz_patterns_corrected_qes_MCva43fa9b_wires.pkl?'
                        '&fmt=pkl'
                        '&pmt_mask=None',
        "s1_optical_propagation_spline": 'itp_map://resource://format://'
                                         'XENONnT_s1_proponly_va43fa9b_wires_20200625.json.gz?'
                                         '&fmt=json.gz'
                                         '&method=RegularGridInterpolator',
        "photon_area_distribution": 'simple_load://resource://format://'
                                   f'{config["photon_area_distribution"]}?'
                                    '&fmt=csv',
        "s2_gain_spread": 0,
        "triplet_lifetime_gas": config["triplet_lifetime_gas"],
        "singlet_lifetime_gas": config["singlet_lifetime_gas"],
        "triplet_lifetime_liquid": config["triplet_lifetime_liquid"],
        "singlet_lifetime_liquid": config["singlet_lifetime_liquid"],
        "singlet_fraction_gas": config["singlet_fraction_gas"],
        "tpc_radius": config["tpc_radius"],
        "diffusion_constant_transverse": config["diffusion_constant_transverse"],
        "s2_aft_skewness": config["s2_aft_skewness"],
        "s2_aft_sigma": config["s2_aft_sigma"],
        "s2_optical_propagation_spline":'itp_map://resource://format://'
                                        'XENONnT_s2_opticalprop_time_v0.json.gz?'
                                        '&fmt=json.gz',
        "s2_luminescence_map":'simple_load://resource://format://'
                              'garfield_timing_map_gas_gap_sr0.npy?'
                              '&fmt=npy',
        "garfield_gas_gap_map": 'itp_map://resource://format://'
                                'garfield_gas_gap_map_sr0.json?'
                                '&fmt=json',
        "pmt_ap_t_modifier": config["pmt_ap_t_modifier"],
        "pmt_ap_modifier": config["pmt_ap_modifier"],
        "photon_ap_cdfs": 'simple_load://resource://format://'
                         f'{config["photon_ap_cdfs"]}?'
                          '&fmt=json.gz',
        "zle_threshold": config["zle_threshold"],
        "digitizer_reference_baseline": config["digitizer_reference_baseline"],
        "enable_noise": config["enable_noise"],
        "high_energy_deamplification_factor": config["high_energy_deamplification_factor"],
        "trigger_window": config["trigger_window"],
        "external_amplification": config["external_amplification"],
        "pmt_pulse_time_rounding": config["pmt_pulse_time_rounding"],
        "samples_after_pulse_center": config["samples_after_pulse_center"],
        "samples_to_store_after": config["samples_to_store_after"],
        "samples_before_pulse_center": config["samples_before_pulse_center"],
        "samples_to_store_before": config["samples_to_store_before"],
        "dt": config["sample_duration"],
        "pe_pulse_ts": config["pe_pulse_ts"],
        "pe_pulse_ys": config["pe_pulse_ys"],
        "rext": 100000,
        "special_thresholds": config["special_thresholds"],
        "noise_data_tmp": 'simple_load://resource://format://'
                         f'{config["noise_file"]}?'
                          '&fmt=npy',

    })

    return st



@URLConfig.register('pmt_gains')
def pmt_gains(to_pe, digitizer_voltage_range, digitizer_bits, pmt_circuit_load_resistor):
    """Build PMT Gains"""
    
    to_pe = to_pe[0][1]
    
    adc_2_current = (digitizer_voltage_range
                     / 2 ** (digitizer_bits)
                     / pmt_circuit_load_resistor)
    
    gains = np.divide(adc_2_current,
                      to_pe,
                      out=np.zeros_like(to_pe),
                      where=to_pe != 0,
                     )
    return gains


@URLConfig.register('pattern_map')
def pattern_map(map_data, pmt_mask, method='WeightedNearestNeighbors'):
    """Pattern map handling"""
    
    if 'compressed' in map_data:
        compressor, dtype, shape = map_data['compressed']
        map_data['map'] = np.frombuffer(
            strax.io.COMPRESSORS[compressor]['decompress'](map_data['map']),
            dtype=dtype).reshape(*shape)
        del map_data['compressed']
    if 'quantized' in map_data:
        map_data['map'] = map_data['quantized']*map_data['map'].astype(np.float32)
        del map_data['quantized']
    if not (pmt_mask is None):
        assert (map_data['map'].shape[-1]==pmt_mask.shape[0]), "Error! Pattern map and PMT gains must have same dimensions!"
        map_data['map'][..., ~pmt_mask]=0.0
    return straxen.InterpolatingMap(map_data, method=method)

#Probably not needed!
@URLConfig.register('simple_load')
def load(data):
    """Some Documentation"""
    return data