# mypy: ignore-errors

from copy import deepcopy
import logging

import numpy as np
import strax
import straxen
from straxen import URLConfig
import fuse

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.context")

# Plugins to simulate microphysics
microphysics_plugins = [
    fuse.micro_physics.ChunkInput,
    fuse.micro_physics.FindCluster,
    fuse.micro_physics.MergeCluster,
    fuse.micro_physics.XENONnT_TPC,
    fuse.micro_physics.XENONnT_BelowCathode,
    fuse.micro_physics.VolumesMerger,
    fuse.micro_physics.ElectricField,
    fuse.micro_physics.NestYields,
    fuse.micro_physics.MicroPhysicsSummary,
]

# Plugins to simulate S1 signals
s1_simulation_plugins = [
    fuse.detector_physics.S1PhotonHits,
    fuse.detector_physics.S1PhotonPropagation,
]

# Plugins to simulate S2 signals
s2_simulation_plugins = [
    fuse.detector_physics.ElectronDrift,
    fuse.detector_physics.ElectronExtraction,
    fuse.detector_physics.ElectronTiming,
    fuse.detector_physics.SecondaryScintillation,
    fuse.detector_physics.S2PhotonPropagation,
]

# Plugins to simulate PMTs and DAQ
pmt_and_daq_plugins = [
    fuse.pmt_and_daq.PMTAfterPulses,
    fuse.pmt_and_daq.PhotonSummary,
    fuse.pmt_and_daq.PulseWindow,
    fuse.pmt_and_daq.PMTResponseAndDAQ,
]

# Plugins to get truth information
truth_information_plugins = [
    fuse.truth_information.RecordsTruth,
    fuse.truth_information.PeakTruth,
    fuse.truth_information.EventTruth,
    fuse.truth_information.SurvivingClusters,
    fuse.truth_information.ClusterTagging,
]


def microphysics_context(
    output_folder="./fuse_data", simulation_config_file="fuse_config_nt_sr1_dev.json"
):
    """Function to create a fuse microphysics simulation context."""

    st = strax.Context(
        storage=strax.DataDirectory(output_folder), **straxen.contexts.xnt_common_opts
    )

    st.config.update(
        dict(
            detector="XENONnT", check_raw_record_overlaps=True, **straxen.contexts.xnt_common_config
        )
    )

    # Register microphysics plugins
    for plugin in microphysics_plugins:
        st.register(plugin)

    set_simulation_config_file(st, simulation_config_file)

    return st


def full_chain_context(
    output_folder="./fuse_data",
    corrections_version=None,
    simulation_config_file="fuse_config_nt_sr1_dev.json",
    corrections_run_id="046477",
    run_id_specific_config={
        "gain_model_mc": "gain_model",
        "electron_lifetime_liquid": "elife",
        "drift_velocity_liquid": "electron_drift_velocity",
        "drift_time_gate": "electron_drift_time_gate",
    },
    run_without_proper_corrections=False,
):
    """Function to create a fuse full chain simulation context."""

    if corrections_run_id is None:
        raise ValueError("Specify a corrections_run_id to load the corrections")
    if (corrections_version is None) & (not run_without_proper_corrections):
        raise ValueError(
            "Specify a corrections_version. If you want to run without proper "
            "corrections for testing or just trying out fuse, "
            "set run_without_proper_corrections to True"
        )
    if simulation_config_file is None:
        raise ValueError("Specify a simulation configuration file")

    if run_without_proper_corrections:
        log.warning(
            "Running without proper correction version. This is not recommended for production use."
            "Take the context defined in cutax if you want to run XENONnT simulations."
        )

    st = strax.Context(
        storage=strax.DataDirectory(output_folder), **straxen.contexts.xnt_common_opts
    )

    st.config.update(
        dict(
            # detector='XENONnT',
            check_raw_record_overlaps=True,
            **straxen.contexts.xnt_common_config,
        )
    )

    # Register microphysics plugins
    for plugin in microphysics_plugins:
        st.register(plugin)

    # Register S1 plugins
    for plugin in s1_simulation_plugins:
        st.register(plugin)

    # Register S2 plugins
    for plugin in s2_simulation_plugins:
        st.register(plugin)

    # Register PMT and DAQ plugins
    for plugin in pmt_and_daq_plugins:
        st.register(plugin)

    # Register truth plugins
    for plugin in truth_information_plugins:
        st.register(plugin)

    if corrections_version is not None:
        st.apply_xedocs_configs(version=corrections_version)

    set_simulation_config_file(st, simulation_config_file)

    local_versions = st.config
    for config_name, url_config in local_versions.items():
        if isinstance(url_config, str):
            if "run_id" in url_config:
                local_versions[config_name] = straxen.URLConfig.format_url_kwargs(
                    url_config, run_id=corrections_run_id
                )
    st.config = local_versions

    # Update some run specific config
    for mc_config, processing_config in run_id_specific_config.items():
        if processing_config in st.config:
            st.config[mc_config] = st.config[processing_config]
        else:
            print(f"Warning! {processing_config} not in context config, skipping...")

    # No blinding in simulations
    st.config["event_info_function"] = "disabled"

    # Deregister plugins with missing dependencies
    st.deregister_plugins_with_missing_dependencies()

    return st


def set_simulation_config_file(context, config_file_name):
    """Function to loop over the plugin config and replace
    SIMULATION_CONFIG_FILE with the actual file name."""
    for data_type, plugin in context._plugin_class_registry.items():
        for option_key, option in plugin.takes_config.items():
            if isinstance(option.default, str) and "SIMULATION_CONFIG_FILE.json" in option.default:
                context.config[option_key] = option.default.replace(
                    "SIMULATION_CONFIG_FILE.json",
                    config_file_name,
                )


@URLConfig.register("pattern_map")
def pattern_map(map_data, pmt_mask, method="WeightedNearestNeighbors"):
    """Pattern map handling."""

    if "compressed" in map_data:
        compressor, dtype, shape = map_data["compressed"]
        map_data["map"] = np.frombuffer(
            strax.io.COMPRESSORS[compressor]["decompress"](map_data["map"]), dtype=dtype
        ).reshape(*shape)
        del map_data["compressed"]
    if "quantized" in map_data:
        map_data["map"] = map_data["quantized"] * map_data["map"].astype(np.float32)
        del map_data["quantized"]
    if not (pmt_mask is None):
        assert (
            map_data["map"].shape[-1] == pmt_mask.shape[0]
        ), "Error! Pattern map and PMT gains must have same dimensions!"
        map_data["map"][..., ~pmt_mask] = 0.0
    return straxen.InterpolatingMap(map_data, method=method)


@URLConfig.register("s2_aft_scaling")
def modify_s2_pattern_map(s2_pattern_map, s2_mean_area_fraction_top, n_tpc_pmts, n_top_pmts):
    """Modify the S2 pattern map to match a given input AFT."""
    if s2_mean_area_fraction_top > 0:
        s2map = deepcopy(s2_pattern_map)
        s2map_topeff_ = s2map.data["map"][..., 0:n_top_pmts].sum(axis=2)
        s2map_toteff_ = s2map.data["map"].sum(axis=2)
        orig_aft_ = np.mean((s2map_topeff_ / s2map_toteff_)[s2map_toteff_ > 0.0])
        # Getting scales for top/bottom separately to preserve total efficiency
        scale_top_ = s2_mean_area_fraction_top / orig_aft_
        scale_bot_ = (1 - s2_mean_area_fraction_top) / (1 - orig_aft_)
        s2map.data["map"][:, :, 0:n_top_pmts] *= scale_top_
        s2map.data["map"][:, :, n_top_pmts:n_tpc_pmts] *= scale_bot_
        s2_pattern_map.__init__(s2map.data)
    return s2_pattern_map


# Probably not needed!
@URLConfig.register("simple_load")
def load(data):
    """Some Documentation."""
    return data


@URLConfig.register("simulation_config")
def from_config(config_name, key):
    """Return a value from a json config file."""
    config = straxen.get_resource(config_name, fmt="json")
    return config[key]
