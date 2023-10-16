import strax
import straxen
import fuse
import numpy as np
from straxen import URLConfig

#Plugins to simulate microphysics
microphysics_plugins = [fuse.micro_physics.ChunkInput,
                        fuse.micro_physics.FindCluster,
                        fuse.micro_physics.MergeCluster,
                        fuse.micro_physics.XENONnT_TPC,
                        fuse.micro_physics.XENONnT_BelowCathode,
                        fuse.micro_physics.VolumesMerger,
                        fuse.micro_physics.ElectricField,
                        fuse.micro_physics.NestYields,
                        fuse.micro_physics.MicroPhysicsSummary]

#Plugins to simulate S1 signals
s1_simulation_plugins = [fuse.detector_physics.S1PhotonHits,
                         fuse.detector_physics.S1PhotonPropagation,
                        ]

#Plugins to simulate S2 signals
s2_simulation_plugins = [fuse.detector_physics.ElectronDrift,
                         fuse.detector_physics.ElectronExtraction,
                         fuse.detector_physics.ElectronTiming,
                         fuse.detector_physics.SecondaryScintillation,
                         fuse.detector_physics.S2PhotonPropagation
                         ]

#Plugins to simulate PMTs and DAQ
pmt_and_daq_plugins = [fuse.pmt_and_daq.PMTAfterPulses,
                       fuse.pmt_and_daq.PhotonSummary,
                       fuse.pmt_and_daq.PulseWindow,
                       fuse.pmt_and_daq.PMTResponseAndDAQ,
                       ]

def microphysics_context(output_folder = "./fuse_data"
                         ):
    """
    Function to create a fuse microphysics simulation context. 
    """

    st = strax.Context(storage=strax.DataDirectory(output_folder),
                       **straxen.contexts.xnt_common_opts)
    
    st.config.update(dict(detector='XENONnT',
                          check_raw_record_overlaps=True,
                          **straxen.contexts.xnt_common_config))
    
    #Register microphysics plugins
    for plugin in microphysics_plugins:
        st.register(plugin)

    return st

def full_chain_context(output_folder = "./fuse_data"
                       ):
    """
    Function to create a fuse full chain simulation context. 
    """

    st = strax.Context(storage=strax.DataDirectory(output_folder),
                       **straxen.contexts.xnt_common_opts)
    
    st.config.update(dict(detector='XENONnT',
                          check_raw_record_overlaps=True,
                          **straxen.contexts.xnt_common_config))

    #Register microphysics plugins
    for plugin in microphysics_plugins:
        st.register(plugin)

    #Register S1 plugins
    for plugin in s1_simulation_plugins:
        st.register(plugin)

    #Register S2 plugins
    for plugin in s2_simulation_plugins:
        st.register(plugin)

    #Register PMT and DAQ plugins
    for plugin in pmt_and_daq_plugins:
        st.register(plugin)

    return st


def set_simulation_config_file(context, config_file_name):
    """
    Function to loop over the plugin config and replace SIMULATION_CONFIG_FILE with the actual file name
    """
    for data_type, plugin in context._plugin_class_registry.items():
        for option_key, option in plugin.takes_config.items():

            if isinstance(option.default, str) and "SIMULATION_CONFIG_FILE" in option.default:
                context.config[option_key] = option.default.replace("SIMULATION_CONFIG_FILE", config_file_name)
                

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

@URLConfig.register('simulation_config')
def from_config(config_name, key):
    """
    Return a value from a json config file
    """
    config = straxen.get_resource(config_name, fmt="json")
    return config[key]