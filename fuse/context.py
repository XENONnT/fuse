# mypy: ignore-errors
import logging
import os
import strax
import straxen
import fuse

from .context_utils import (
    write_run_id_to_config,
    set_simulation_config_file,
    old_xedocs_versions_patch,
    overwrite_map_from_config,
    apply_mc_overrides,
)

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.context")

DEFAULT_XEDOCS_VERSION = "global_v16"
DEFAULT_SIMULATION_VERSION = "sr1_dev"

# Determine which config names to use (backward compatibility)
if hasattr(straxen.contexts, "xnt_common_opts"):
    # This is for straxen <=2
    common_opts = straxen.contexts.xnt_common_opts
    common_config = straxen.contexts.xnt_common_config
else:
    # This is for straxen >=3, variable names changed
    common_opts = straxen.contexts.common_opts
    common_config = straxen.contexts.common_config


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

    st = strax.Context(storage=output_folder, **common_opts)

    st.config.update(dict(detector="XENONnT", check_raw_record_overlaps=True, **common_config))

    # Register microphysics plugins
    for plugin in microphysics_plugins_dbscan_clustering:
        st.register(plugin)
    for plugin in remaining_microphysics_plugins:
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
    run_without_proper_run_id=False,
    clustering_method="dbscan",
    extra_plugins=[],
):
    """Function to create a fuse full chain simulation context."""

    # Lets go for info level logging when working with fuse
    log.setLevel("INFO")

    if (corrections_run_id is None) & (not run_without_proper_run_id):
        raise ValueError(
            "Specify a corrections_run_id. If you want to run without proper "
            "run_id for testing or just trying out fuse, "
            "set run_without_proper_run_id to True"
        )
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

    st = strax.Context(storage=output_folder, **common_opts)
    st.simulation_config_file = simulation_config_file
    st.corrections_run_id = corrections_run_id

    st.config.update(dict(check_raw_record_overlaps=True, **common_config))  # detector='XENONnT',

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

    # Register extra plugins
    n_extra = len(extra_plugins)
    if n_extra > 0:
        log.info(f"Registering {n_extra} extra plugins:")
        for plugin in extra_plugins:
            log.info(f"{plugin}")
            st.register(plugin)

    if corrections_version is not None:
        st.apply_xedocs_configs(version=corrections_version)

    set_simulation_config_file(st, simulation_config_file)

    if not run_without_proper_run_id:
        write_run_id_to_config(st, corrections_run_id)

    # Update some run specific config
    for mc_config, processing_config in run_id_specific_config.items():
        if processing_config in st.config:
            st.config[mc_config] = st.config[processing_config]
        else:
            print(f"Warning! {processing_config} not in context config, skipping...")

    # No blinding in simulations
    st.set_config({"event_info_function": "disabled"})

    # Deregister plugins with missing dependencies
    st.deregister_plugins_with_missing_dependencies()

    # Purge unused configs (only for newer strax)
    if hasattr(st, "purge_unused_configs"):
        st.purge_unused_configs()

    return st


def xenonnt_fuse_full_chain_simulation(
    output_folder="./fuse_data",
    corrections_version=DEFAULT_XEDOCS_VERSION,
    simulation_config=DEFAULT_SIMULATION_VERSION,
    corrections_run_id=None,
    run_without_proper_run_id=False,
    clustering_method=None,  # defaults to dbscan, but can be set to lineage
    cut_list=None,
    **kwargs,
):
    """Function to create a fuse full chain simulation context with the proper
    settings for XENONnT simulations.

    It takes the general full_chain_context and sets the proper
    corrections and configuration files for XENONnT.
    """

    # Lets go for info level logging when working with fuse
    log.setLevel("INFO")

    # Check if the provided simulation_config is a file path
    if os.path.isfile(simulation_config):
        simulation_config_file = simulation_config
    else:
        # Get the simulation config file from private_nt_aux_files
        simulation_config_file = "fuse_config_nt_{:s}.json".format(simulation_config)
    log.info(f"Using simulation config file: {simulation_config_file}")

    # Get the corrections_run_id from argument or from config file
    if not run_without_proper_run_id:
        corrections_run_id = corrections_run_id or fuse.from_config(
            simulation_config_file, "default_corrections_run_id"
        )
        log.info(f"Using corrections_run_id: {corrections_run_id}")

    # Get clustering method
    # if it is specified as an argument, use that
    # if it is not specified, try to get it from the config file
    # if it is not in the config file, use dbscan
    if clustering_method is None:
        try:
            clustering_method = fuse.from_config(simulation_config_file, "clustering_method")
        except ValueError:
            clustering_method = "dbscan"
    log.info(f"Using clustering method: {clustering_method}")

    st = fuse.full_chain_context(
        output_folder=output_folder,
        corrections_version=corrections_version,
        simulation_config_file=simulation_config_file,
        corrections_run_id=corrections_run_id,
        run_without_proper_run_id=run_without_proper_run_id,
        clustering_method=clustering_method,
        **kwargs,
    )
    st.set_config(old_xedocs_versions_patch(corrections_version))

    # Load the full config once
    config = straxen.get_resource(simulation_config_file, fmt="json")

    # If mc_overrides is present, use that exclusively
    if "mc_overrides" in config:
        log.info("Found 'mc_overrides' in config,  using override-based config system.")
        apply_mc_overrides(st, simulation_config_file)
    else:
        # Backward compatibility: legacy fdc_map_mc logic
        if "fdc_map_mc" in config:
            fdc_map_mc = config["fdc_map_mc"]
            log.info(f"[legacy] Using fdc_map_mc: {fdc_map_mc}")
            fdc_ext = fdc_map_mc.split(fdc_map_mc.split(".")[0] + ".")[-1]
            fdc_conf = f"itp_map://resource://{fdc_map_mc}?fmt={fdc_ext}"
            st.set_config({"fdc_map": fdc_conf})
        else:
            raise RuntimeError(
                "No 'mc_overrides' or 'fdc_map_mc' found in the config. "
                "Please define one of them to set 'fdc_map'."
            )

    if cut_list is not None:
        st.register_cut_list(cut_list)

    return st


def public_config_context(
    output_folder="./fuse_data",
    extra_plugins=[fuse.plugins.S2PhotonPropagationSimple],
    simulation_config_file="./files/XENONnT_public_config.json",
):
    """Function to create a fuse full chain simulation context."""

    # Lets go for info level logging when working with fuse
    log.setLevel("INFO")

    if simulation_config_file is None:
        raise ValueError("Specify a simulation configuration file")

    st = strax.Context(storage=output_folder, **straxen.contexts.common_opts)
    st.simulation_config_file = simulation_config_file
    st.config.update(dict(check_raw_record_overlaps=True, **straxen.contexts.common_config))

    # Register microphysics plugins
    for plugin in microphysics_plugins_dbscan_clustering:
        st.register(plugin)

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

    # Register extra plugins
    n_extra = len(extra_plugins)
    if n_extra > 0:
        log.info(f"Registering {n_extra} extra plugins:")
        for plugin in extra_plugins:
            log.info(f"{plugin}")
            st.register(plugin)

    set_simulation_config_file(st, simulation_config_file)

    # Lets override some resource files with the ones from the simulation config
    config = straxen.get_resource(simulation_config_file, fmt="json")
    overwrite_map_from_config(st, config)

    # And finally some hardcoded configs
    st.set_config({"s1_lce_correction_map": "constant_dummy_map://1"})
    st.set_config(
        {
            "gain_model_mc": "simple_load://resource://./files/fake_to_pe.npy?&fmt=npy",
        }
    )

    # No blinding in simulations
    st.set_config({"event_info_function": "disabled"})

    # Deregister plugins with missing dependencies
    st.deregister_plugins_with_missing_dependencies()

    # Purge unused configs
    st.purge_unused_configs()

    return st
