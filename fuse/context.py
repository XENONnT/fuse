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
log.setLevel("INFO")

# Backward compatibility for straxen versions
if hasattr(straxen.contexts, "xnt_common_opts"):
    common_opts = straxen.contexts.xnt_common_opts
    common_config = straxen.contexts.xnt_common_config
else:
    common_opts = straxen.contexts.common_opts
    common_config = straxen.contexts.common_config


# Plugins to simulate microphysics
microphysics_plugins_clustering = {
    "dbscan": [
        fuse.micro_physics.ChunkInput,
        fuse.micro_physics.FindCluster,
        fuse.micro_physics.MergeCluster,
    ],
    "lineage": [
        fuse.micro_physics.ChunkInput,
        fuse.micro_physics.LineageClustering,
        fuse.micro_physics.MergeLineage,
    ],
}

# Plugins to simulate microphysics (remaining)
microphysics_plugins_remaining = [
    fuse.micro_physics.cuts_and_selections.VolumeSelection,
    fuse.micro_physics.cuts_and_selections.DefaultSimulation,
    fuse.micro_physics.ElectricField,
    fuse.micro_physics.NestYields,
    fuse.micro_physics.MicroPhysicsSummary,
    fuse.micro_physics.VolumeProperties,
]

# Plugins to simulate S1 signals
s1_simulation_plugins = [
    fuse.detector_physics.S1PhotonHits,
    fuse.detector_physics.S1PhotonPropagation,
]

# Plugins to simulate S2 signals
s2_simulation_plugins = [
    fuse.detector_physics.ElectronDrift,
    fuse.detector_physics.ElectronPropagation,
    fuse.detector_physics.ElectronExtraction,
    fuse.detector_physics.SecondaryScintillation,
    fuse.detector_physics.S2PhotonPropagation,
]

# Plugins to simulate delayed electrons
delayed_electron_simulation_plugins = [
    fuse.detector_physics.delayed_electrons.PhotoIonizationElectrons,
    fuse.detector_physics.delayed_electrons.DelayedElectronsDrift,
    fuse.detector_physics.delayed_electrons.DelayedElectronPropagation,
    fuse.detector_physics.delayed_electrons.DelayedElectronsExtraction,
    fuse.detector_physics.delayed_electrons.DelayedElectronsSecondaryScintillation,
    fuse.detector_physics.delayed_electrons.S1PhotonHitsEmpty,
]

# Plugins to merge delayed and regular electrons
delayed_electron_merger_plugins = [
    fuse.detector_physics.delayed_electrons.DriftedElectronsMerger,
    fuse.detector_physics.delayed_electrons.PropagatedElectronsMerger,
    fuse.detector_physics.delayed_electrons.ExtractedElectronsMerger,
    fuse.detector_physics.delayed_electrons.SecondaryScintillationPhotonsMerger,
    fuse.detector_physics.delayed_electrons.SecondaryScintillationPhotonSumMerger,
    fuse.detector_physics.delayed_electrons.MicrophysicsSummaryMerger,
    fuse.detector_physics.delayed_electrons.S1PhotonHitsMerger,
]

perpendicular_wire_shift_plugins = [
    fuse.detector_physics.ElectronPropagationPerpWires,
    fuse.detector_physics.delayed_electrons.DelayedElectronPropagationPerpWires,
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
processing_plugins = [fuse.processing.CorrectedAreasMC]


def microphysics_context(
    output_folder="./fuse_data",
    simulation_config_file="fuse_config_nt_sr1_dev.json",
    clustering_method="dbscan",
    extra_plugins=[],
):
    """Function to create a fuse microphysics simulation context."""

    st = strax.Context(storage=output_folder, **common_opts)
    st.set_config(dict(check_raw_record_overlaps=True, **common_config))

    # Register microphysics plugins
    for plugin_list in [
        microphysics_plugins_clustering[clustering_method],
        microphysics_plugins_remaining,
        extra_plugins,
    ]:
        for p in plugin_list:
            st.register(p)

    set_simulation_config_file(st, simulation_config_file)

    return st


def xenonnt_fuse_full_chain_simulation(
    output_folder="./fuse_data",
    corrections_version=None,
    simulation_config=None,
    corrections_run_id=None,
    clustering_method=None,
    cut_list=None,
    extra_plugins=[],
    run_id_specific_config={
        "gain_model_mc": "gain_model",
        "electron_lifetime_liquid": "elife",
        "drift_velocity_liquid": "electron_drift_velocity",
        "drift_time_gate": "electron_drift_time_gate",
    },
    run_without_proper_run_id=False,
    run_without_config_file=False,
):
    """Create a context for the full chain simulation of XENONnT.

    This context includes all the necessary configs and plugins for the
    simulation.
    """

    # Load config file
    if run_without_config_file:
        # Just a dummy name to avoid errors. We use this to setup context for the docs.
        simulation_config_file = "fuse_config_file.json"
        sim_config = {
            "fdc_map_mc": ".",
        }
        log.warning("Running without a proper config file! Do not use this for real simulations.")
    else:
        if simulation_config:
            simulation_config_file = (
                simulation_config
                if os.path.isfile(simulation_config)
                else f"fuse_config_nt_{simulation_config}.json"
            )
            sim_config = straxen.get_resource(simulation_config_file, fmt="json")
        else:
            raise ValueError(
                "simulation_config_file is required. "
                "Please provide a valid file path or file name."
            )

    # Load settings from config file
    if not run_without_proper_run_id:
        corrections_run_id = (
            corrections_run_id
            if corrections_run_id is not None
            else sim_config.get("default_corrections_run_id", None)
        )
        if corrections_run_id is None:
            raise ValueError(
                "corrections_run_id is required. Please provide a config file with "
                "default_corrections_run_id or provide it directly in the context function."
            )
        log.info(f"Using corrections run id: {corrections_run_id}")

    clustering_method = (
        clustering_method
        if clustering_method is not None
        else sim_config.get("clustering_method", "dbscan")
    )
    log.info(f"Using clustering method: {clustering_method}")

    enable_perp_wire_electron_shift = sim_config.get("enable_perp_wire_electron_shift", False)

    # Create context and register plugins
    st = strax.Context(storage=output_folder, **common_opts)
    st.set_config(dict(check_raw_record_overlaps=True, **common_config))
    st.simulation_config_file = simulation_config_file
    st.corrections_run_id = corrections_run_id

    if any("cutax" in str(p) for p in extra_plugins) or cut_list:
        import cutax

        extra_plugins.extend(cutax.EXTRA_PLUGINS)

    # Register all plugins
    # The order here matters!
    plugin_lists = [
        microphysics_plugins_clustering[clustering_method],
        microphysics_plugins_remaining,
        s1_simulation_plugins,
        s2_simulation_plugins,
        delayed_electron_simulation_plugins,
        delayed_electron_merger_plugins,
        pmt_and_daq_plugins,
        truth_information_plugins,
        processing_plugins,
    ]

    # Perpendicular wire electron shift plugin
    if enable_perp_wire_electron_shift:
        log.info("Enabling perpendicular wire electron shift plugin.")
        plugin_lists.append(perpendicular_wire_shift_plugins)

    # Lastly, any extra plugins the user wants to add
    # Needs to be last, in case it overrides any of the above
    plugin_lists.append(extra_plugins)

    for plugin_list in plugin_lists:
        for plugin in plugin_list:
            st.register(plugin)

    if cut_list:
        st.register_cut_list(cut_list)
        st.register(cut_list)

    # Corrections setup
    if corrections_version:
        st.apply_xedocs_configs(version=corrections_version)
        st.set_config(old_xedocs_versions_patch(corrections_version))
    else:
        log.warning(
            "Running without proper corrections! Please provide a corrections_version "
            "to ensure proper corrections. "
            "Example: 'global_v16'"
        )

    # Replace SIMULATION_CONFIG_FILE.json in plugin defaults
    if simulation_config_file:
        set_simulation_config_file(st, simulation_config_file)

    if not run_without_proper_run_id:
        write_run_id_to_config(st, corrections_run_id)

    # Update some run specific config
    for mc_config, processing_config in run_id_specific_config.items():
        if processing_config in st.config:
            st.config[mc_config] = st.config[processing_config]
        else:
            log.warning(f"{processing_config} not in context config, skipping...")

    # If mc_overrides is present, use that exclusively
    if "mc_overrides" in sim_config:
        log.info("Found 'mc_overrides' in config,  using override-based config system.")
        apply_mc_overrides(st, simulation_config_file)
    else:
        # Backward compatibility: legacy fdc_map_mc logic
        if "fdc_map_mc" in sim_config:
            fdc_map_mc = sim_config["fdc_map_mc"]
            log.info(f"[legacy] Using fdc_map_mc: {fdc_map_mc}")
            fdc_ext = fdc_map_mc.split(fdc_map_mc.split(".")[0] + ".")[-1]
            fdc_conf = f"itp_map://resource://{fdc_map_mc}?fmt={fdc_ext}"
            st.set_config({"fdc_map": fdc_conf})
        else:
            raise RuntimeError(
                "No 'mc_overrides' or 'fdc_map_mc' found in the config. "
                "Please define one of them to set 'fdc_map'."
            )

    # No blinding in simulations
    st.set_config({"event_info_function": "disabled"})

    st.deregister_plugins_with_missing_dependencies()

    if hasattr(st, "purge_unused_configs"):
        st.purge_unused_configs()

    return st


def public_config_context(
    output_folder="./fuse_data",
    extra_plugins=[fuse.plugins.S2PhotonPropagationSimple],
    simulation_config_file="./files/XENONnT_public_config.json",
    clustering_method="dbscan",
):
    """Create a context for the use of fuse with public XENONnT configs."""

    st = strax.Context(storage=output_folder, **straxen.contexts.common_opts)
    st.simulation_config_file = simulation_config_file
    st.config.update(dict(check_raw_record_overlaps=True, **straxen.contexts.common_config))

    for plugin_list in [
        microphysics_plugins_clustering[clustering_method],
        microphysics_plugins_remaining,
        s1_simulation_plugins,
        s2_simulation_plugins,
        delayed_electron_simulation_plugins,
        delayed_electron_merger_plugins,
        pmt_and_daq_plugins,
        truth_information_plugins,
        processing_plugins,
        extra_plugins,
    ]:
        for plugin in plugin_list:
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

    # Remove unused configs
    if hasattr(st, "purge_unused_configs"):
        st.purge_unused_configs()

    return st
