import strax
import straxen

from ...dtypes import (
    primary_positions_fields,
    cluster_positions_fields,
    cluster_id_fields,
    cluster_misc_fields,
)
from ...volume_plugin import VolumePlugin
from ...vertical_merger_plugin import VerticalMergerPlugin


class VolumesMerger(VerticalMergerPlugin):
    """Plugin that concatenates the clusters that are in the XENONnT TPC or the
    volume below the cathode."""

    depends_on = ("tpc_interactions", "below_cathode_interactions")

    provides = "interactions_in_roi"
    data_kind = "interactions_in_roi"
    __version__ = "0.2.0"

    def compute(self, **kwargs):
        return super().compute(**kwargs)


# Fixed detector dimensions of XENONnT:
# See also: https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:analysis:coordinate_system


class XENONnT_TPC(VolumePlugin):
    """Plugin to select only clusters in the XENONnT TPC.

    The TPC volume
    is defined by the z position of the cathode and gate mesh and by the radius
    of the detector. For all clusters passing the volume selection ``create_S2`` is set
    to ``True``.
    """

    __version__ = "0.3.2"
    depends_on = "clustered_interactions"
    provides = "tpc_interactions"
    data_kind = "tpc_interactions"

    dtype = (
        cluster_positions_fields
        + cluster_id_fields
        + cluster_misc_fields
        + primary_positions_fields
        + strax.time_fields
    )

    # Config options
    # Define the TPC volume
    xenonnt_z_cathode = straxen.URLConfig(
        default=-148.6515,  # Top of the cathode electrode
        type=(int, float),
        help="z position of the XENONnT cathode [cm]",
    )

    xenonnt_z_gate_mesh = straxen.URLConfig(
        default=0.0,  # bottom of the gate electrode
        type=(int, float),
        help="z position of the XENONnT gate mesh [cm]",
    )

    xenonnt_sensitive_volume_radius = straxen.URLConfig(
        default=66.4,
        type=(int, float),
        help="Radius of the XENONnT TPC [cm]",
    )

    xenon_density_tpc = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=xenon_density_tpc",
        type=(int, float),
        help="Density of xenon in the TPC volume [g/cm3]",
    )

    create_S2_xenonnt_TPC = straxen.URLConfig(
        default=True,
        type=bool,
        help="Create S2s in the XENONnT TPC",
    )

    def compute(self, clustered_interactions):
        # Call the ROI function:
        mask = self.in_ROI(
            clustered_interactions,
            self.xenonnt_z_cathode,
            self.xenonnt_z_gate_mesh,
            self.xenonnt_sensitive_volume_radius,
        )

        tpc_interactions = clustered_interactions[mask]
        tpc_interactions["xe_density"] = self.xenon_density_tpc
        tpc_interactions["vol_id"] = 1  # Do we need this? -> Now just copied from epix
        tpc_interactions["create_S2"] = self.create_S2_xenonnt_TPC

        return tpc_interactions


class XENONnT_BelowCathode(VolumePlugin):
    """Plugin to select only clusters  below the XENONnT cathode.

    The volume
    is defined by the z position of the cathode and bottom PMTs and by the radius
    of the detector. For all clusters passing the volume selection ``create_S2`` is set
    to ``False``.
    """

    __version__ = "0.3.2"
    depends_on = "clustered_interactions"
    provides = "below_cathode_interactions"
    data_kind = "below_cathode_interactions"

    dtype = (
        cluster_positions_fields
        + cluster_id_fields
        + cluster_misc_fields
        + primary_positions_fields
        + strax.time_fields
    )

    # Config options
    # Define the volume
    xenonnt_z_cathode = straxen.URLConfig(
        default=-148.6515,  # Top of the cathode electrode
        type=(int, float),
        help="z position of the XENONnT cathode [cm]",
    )

    xenonnt_z_bottom_pmts = straxen.URLConfig(
        default=-154.6555,  # Top surface of the bottom PMT window
        type=(int, float),
        help="z position of the XENONnT bottom PMT array [cm]",
    )

    xenonnt_sensitive_volume_radius = straxen.URLConfig(
        default=66.4,
        type=(int, float),
        help="Radius of the XENONnT TPC [cm]",
    )

    xenon_density_below_cathode = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=xenon_density_tpc",
        type=(int, float),
        help="Density of xenon in the below-cathode-volume [g/cm3]",
    )

    create_S2_xenonnt_below_cathode = straxen.URLConfig(
        default=False,
        type=bool,
        help="No S2s from below the cathode",
    )

    def compute(self, clustered_interactions):
        # Call the ROI function:
        mask = self.in_ROI(
            clustered_interactions,
            self.xenonnt_z_bottom_pmts,
            self.xenonnt_z_cathode,
            self.xenonnt_sensitive_volume_radius,
        )

        tpc_interactions = clustered_interactions[mask]
        tpc_interactions["xe_density"] = self.xenon_density_below_cathode
        tpc_interactions["vol_id"] = 2  # Do we need this? -> Now just copied from epix
        tpc_interactions["create_S2"] = self.create_S2_xenonnt_below_cathode

        return tpc_interactions


class XENONnT_GasPhase(VolumePlugin):
    """Plugin that evaluates if interactions are in the gas phase of XENONnT.

    Only these interactions are returned. The output of this plugin is
    meant to go into a vertical merger plugin.
    """

    __version__ = "0.3.2"
    depends_on = "clustered_interactions"
    provides = "gas_phase_interactions"
    data_kind = "gas_phase_interactions"

    dtype = (
        cluster_positions_fields
        + cluster_id_fields
        + cluster_misc_fields
        + primary_positions_fields
        + strax.time_fields
    )

    # Config options
    # Define the volume
    xenonnt_z_top_pmts = straxen.URLConfig(
        default=7.3936,  # cm
        type=(int, float),
        help="Position of the top of gas phase [cm]",
    )

    xenonnt_z_lxe = straxen.URLConfig(
        default=0.416,  # cm ... liquid-gas interface
        type=(int, float),
        help="Position of the bottom of gas phase [cm]",
    )

    xenonnt_sensitive_volume_radius = straxen.URLConfig(
        default=66.4,  # cm
        type=(int, float),
        help="Radius of the XENONnT TPC [cm]",
    )

    xenon_density_gas_phase = straxen.URLConfig(
        default=0.0177,
        type=(int, float),
        help="Density of XENON in the gas phase [g/cm3]",
    )

    create_S2_xenonnt_gas_phase = straxen.URLConfig(
        default=False,
        type=bool,
        help="Whether generate S2s in gas phase",
    )

    def compute(self, clustered_interactions):
        # Call the ROI function:
        mask = self.in_ROI(
            clustered_interactions,
            self.xenonnt_z_lxe,
            self.xenonnt_z_top_pmts,
            self.xenonnt_sensitive_volume_radius,
        )

        tpc_interactions = clustered_interactions[mask]
        tpc_interactions["xe_density"] = self.xenon_density_gas_phase
        tpc_interactions["vol_id"] = 3  # Do we need this? -> Now just copied from epix
        tpc_interactions["create_S2"] = self.create_S2_xenonnt_gas_phase

        return tpc_interactions
