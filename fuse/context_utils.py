import numpy as np
import strax
import straxen
from straxen import URLConfig
from copy import deepcopy
import logging

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.context_utils")


def set_run_id_config(context, key, value, corrections_run_id):
    """Set run_id of a URLConfig in the context config."""
    context.set_config({key: straxen.URLConfig.format_url_kwargs(value, run_id=corrections_run_id)})


def replace_run_id_config(context, key, value, corrections_run_id):
    """Set run_id of a URLConfig in the context config."""
    context.set_config({key: value.replace("plugin.run_id", corrections_run_id)})


def write_run_id_to_config(context, corrections_run_id):
    """Overwrite any run_id dependent URLConfig in the config."""
    for pattern, function in zip(
        ["run_id=", "science_run://"], [set_run_id_config, replace_run_id_config]
    ):
        for config_name, url_config in context.config.copy().items():
            if isinstance(url_config, str) and pattern in url_config:
                function(context, config_name, url_config, corrections_run_id)
    # Actually this is not needed, but it is here for completeness
    # Usually configs contains science_run:// will not be changed when applying global_version,
    # so we decorate science_run:// by the default of plugin configs
    # Because global_version guaranteed that all time dependent configs are assigned,
    # we do not have to decorate run_id= by the default of plugin configs
    for pattern, function in zip(["science_run://"], [replace_run_id_config]):
        for data_type, plugin in context._plugin_class_registry.items():
            for option_key, option in plugin.takes_config.items():
                if isinstance(option.default, str) and pattern in option.default:
                    function(context, option_key, option.default, corrections_run_id)


def old_xedocs_versions_patch(xedocs_version):
    """To ensure backward compatibility for global version < 15 If a user still
    assigns global_v14 or smaller, the avg_se_gain, g1, and g2 are taken from
    BODEGA."""
    if int(xedocs_version.replace("global_v", "")) < 15:
        return {
            "avg_se_gain": "bodega://se_gain?bodega_version=v1",
            "g1": "bodega://g1?bodega_version=v5",
            "g2": "bodega://g2?bodega_version=v5",
        }
    else:
        return {}


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
    """Load a key from a simulation config file."""
    config = straxen.get_resource(config_name, fmt="json")
    if key not in config:
        raise ValueError(f"Key {key} not found in {config_name}")
    return config[key]


class DummyMap:
    """Return constant results with length equal to that of the input and
    second dimensions (constand correction) user-defined."""

    def __init__(self, const, shape=()):
        self.const = float(const)
        self.shape = shape
        self.data = {"map": np.ones(shape)}

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


def overwrite_map_from_config(
    context,
    config,
    resource_keys=[
        "efield_map",
        "s1_time_spline",
        "s1_pattern_map",
        "s2_time_spline",
        "s2_pattern_map",
        "s2_correction_map",
        "photon_area_distribution",
        "photon_ap_cdfs",
        "noise_file",
    ],
):
    for key, value in context.config.items():
        if isinstance(value, str):
            matching_keys = np.array(
                ["=" + resource_key in value for resource_key in resource_keys]
            )

            if any(matching_keys):
                context.set_config({key: config[resource_keys[np.argwhere(matching_keys)[0][0]]]})


@URLConfig.register("lce_from_pattern_map")
def lce_from_pattern_map(map, pmt_mask):
    """Build a S1 lce correction map from a S1 pattern map."""

    lcemap = deepcopy(map)
    lcemap.data["map"] = np.sum(lcemap.data["map"][:][:][:], axis=3, keepdims=True, where=pmt_mask)
    lcemap.__init__(lcemap.data)
    return lcemap


def apply_mc_overrides(context, config_file):
    """Apply config overrides from 'mc_overrides' using from_config."""
    try:
        config = straxen.get_resource(config_file, fmt="json")
        overrides = config.get("mc_overrides", {})
        for key, value in overrides.items():
            if isinstance(value, list) and len(value) == 2:
                filename, template = value
                url = template.replace("MAP_NAME", filename)
            else:
                url = value
            context.set_config({key: url})
            log.debug(f"[mc_overrides] Set '{key}' to '{value}'")
    except Exception as e:
        raise ValueError(f"[mc_overrides] Failed to apply overrides from {config_file}: {e}") from e
