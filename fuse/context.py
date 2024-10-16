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
microphysics_plugins_dbscan_clustering = [
    fuse.micro_physics.ChunkInput,
    fuse.micro_physics.FindCluster,
    fuse.micro_physics.MergeCluster,
]

microphysics_plugins_lineage_clustering = [
    fuse.micro_physics.ChunkInput,
    fuse.micro_physics.LineageClustering,
    fuse.micro_physics.MergeLineage,
]

remaining_microphysics_plugins = [
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

# Plugins to simulate delayed electrons
delayed_electron_simulation_plugins = [
    fuse.detector_physics.delayed_electrons.PhotoIonizationElectrons,
    fuse.detector_physics.delayed_electrons.DelayedElectronsDrift,
    fuse.detector_physics.delayed_electrons.DelayedElectronsExtraction,
    fuse.detector_physics.delayed_electrons.DelayedElectronsTiming,
    fuse.detector_physics.delayed_electrons.DelayedElectronsSecondaryScintillation,
    fuse.detector_physics.delayed_electrons.S1PhotonHitsEmpty,
]

# Plugins to merge delayed and regular electrons
delayed_electron_merger_plugins = [
    fuse.detector_physics.delayed_electrons.DriftedElectronsMerger,
    fuse.detector_physics.delayed_electrons.ExtractedElectronsMerger,
    fuse.detector_physics.delayed_electrons.ElectronTimingMerger,
    fuse.detector_physics.delayed_electrons.SecondaryScintillationPhotonsMerger,
    fuse.detector_physics.delayed_electrons.SecondaryScintillationPhotonSumMerger,
    fuse.detector_physics.delayed_electrons.MicrophysicsSummaryMerger,
    fuse.detector_physics.delayed_electrons.S1PhotonHitsMerger,
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

# Plugins to override the default processing plugins in straxen
processing_plugins = [
    fuse.processing.CorrectedAreasMC,
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
    for plugin in microphysics_plugins_dbscan_clustering:
        st.register(plugin)
    for plugin in remaining_microphysics_plugins:
        st.register(plugin)

    set_simulation_config_file(st, simulation_config_file)

    return st


def full_chain_context(
    output_folder="./fuse_data",
    clustering_method="dbscan",
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

    # Lets go for info level logging when working with fuse
    log.setLevel("INFO")

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
        dict(  # detector='XENONnT',
            check_raw_record_overlaps=True, **straxen.contexts.xnt_common_config
        )
    )

    # Register microphysics plugins
    if clustering_method == "dbscan":
        for plugin in microphysics_plugins_dbscan_clustering:
            st.register(plugin)
    elif clustering_method == "lineage":
        for plugin in microphysics_plugins_lineage_clustering:
            st.register(plugin)
    else:
        raise ValueError(f"Clustering method {clustering_method} not implemented!")

    for plugin in remaining_microphysics_plugins:
        st.register(plugin)

    # Register S1 plugins
    for plugin in s1_simulation_plugins:
        st.register(plugin)

    # Register S2 plugins
    for plugin in s2_simulation_plugins:
        st.register(plugin)

    # Register delayed Electrons plugins
    for plugin in delayed_electron_simulation_plugins:
        st.register(plugin)

    # Register merger plugins.
    for plugin in delayed_electron_merger_plugins:
        st.register(plugin)

    # Register PMT and DAQ plugins
    for plugin in pmt_and_daq_plugins:
        st.register(plugin)

    # Register truth plugins
    for plugin in truth_information_plugins:
        st.register(plugin)

    # Register processing plugins
    log.info("Overriding processing plugins:")
    for plugin in processing_plugins:
        log.info(f"Registering {plugin}")
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

    # Purge unused configs
    st.purge_unused_configs()

    return st


def set_simulation_config_file(context, config_file_name):
    """Function to loop over the plugin config and replace
    SIMULATION_CONFIG_FILE with the actual file name."""
    for data_type, plugin in context._plugin_class_registry.items():
        for option_key, option in plugin.takes_config.items():
            if isinstance(option.default, str) and "SIMULATION_CONFIG_FILE.json" in option.default:
                context.config[option_key] = option.default.replace(
                    "SIMULATION_CONFIG_FILE.json", config_file_name
                )

            # Special case for the photoionization_modifier
            if option_key == "photoionization_modifier":
                context.config[option_key] = option.default


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
def modify_s2_pattern_map(
    s2_pattern_map, s2_mean_area_fraction_top, n_tpc_pmts, n_top_pmts, turned_off_pmts
):
    """Modify the S2 pattern map to match a given input AFT."""
    if s2_mean_area_fraction_top > 0:
        s2map = deepcopy(s2_pattern_map)
        # First we need to set turned off pmts before scaling
        s2map.data["map"][..., turned_off_pmts] = 0
        s2map_topeff_ = s2map.data["map"][..., :n_top_pmts].sum(axis=2, keepdims=True)
        s2map_toteff_ = s2map.data["map"].sum(axis=2, keepdims=True)
        orig_aft_ = np.nanmean(s2map_topeff_ / s2map_toteff_)
        # Getting scales for top/bottom separately to preserve total efficiency
        scale_top_ = s2_mean_area_fraction_top / orig_aft_
        scale_bot_ = (1 - s2_mean_area_fraction_top) / (1 - orig_aft_)
        s2map.data["map"][..., :n_top_pmts] *= scale_top_
        s2map.data["map"][..., n_top_pmts:n_tpc_pmts] *= scale_bot_
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


class DummyMap:
    """Return constant results with length equal to that of the input and
    second dimensions (constand correction) user-defined."""

    def __init__(self, const, shape=()):
        self.const = float(const)
        self.shape = shape

    def __call__(self, x, **kwargs):
        shape = [len(x)] + list(self.shape)
        return np.ones(shape) * self.const

    def reduce_last_dim(self):
        assert len(self.shape) >= 1, "Need at least 1 dim to reduce further"
        const = self.const * self.shape[-1]
        shape = list(self.shape)
        shape[-1] = 1

        return DummyMap(const, shape)


@URLConfig.register("constant_dummy_map")
def get_dummy(const, shape=()):
    """Make an Dummy Map."""
    itp_map = DummyMap(const, shape)
    return itp_map
