import strax
import straxen
import numba

import numpy as np

from ...dtypes import (
    volume_properties_fields,
)

from ...common import VOLUMES_IDS
from ...plugin import FuseBasePlugin

# Fixed detector dimensions of XENONnT:
# See also: https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:analysis:coordinate_system


class VolumeProperties(FuseBasePlugin):
    """Plugin that concatenates the clusters that are in the XENONnT TPC or the
    volume below the cathode."""

    depends_on = "clustered_interactions"
    provides = "volume_properties"

    __version__ = "0.0.5"

    dtype = volume_properties_fields + strax.time_fields

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

    xenonnt_z_top_pmts = straxen.URLConfig(
        default=7.3936,  # cm
        type=(int, float),
        help="Position of the top of gas phase [cm]",
    )

    xenonnt_z_lxe = straxen.URLConfig(
        default="take://resource://SIMULATION_CONFIG_FILE.json?&fmt=json&take=xenonnt_z_lxe",
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

    def in_ROI(self, interactions, min_z, max_z, max_r):
        """Function that evaluates if an interaction is in the ROI."""
        mask = in_cylinder(
            interactions["x"], interactions["y"], interactions["z"], min_z, max_z, max_r
        )
        return mask

    def compute(self, clustered_interactions):

        if len(clustered_interactions) == 0:
            return np.zeros(0, dtype=self.dtype)

        result = np.zeros(len(clustered_interactions), dtype=self.dtype)
        result["time"] = clustered_interactions["time"]
        result["endtime"] = clustered_interactions["endtime"]

        # TPC Volume
        mask_tpc = self.in_ROI(
            clustered_interactions,
            self.xenonnt_z_cathode,
            self.xenonnt_z_gate_mesh,
            self.xenonnt_sensitive_volume_radius,
        )

        # Below Cathode Volume
        mask_below_cathode = self.in_ROI(
            clustered_interactions,
            self.xenonnt_z_bottom_pmts,
            self.xenonnt_z_cathode,
            self.xenonnt_sensitive_volume_radius,
        )

        # Gas Phase Volume
        mask_gas_phase = self.in_ROI(
            clustered_interactions,
            self.xenonnt_z_lxe,
            self.xenonnt_z_top_pmts,
            self.xenonnt_sensitive_volume_radius,
        )

        if np.any(mask_tpc):
            result["vol_id"][mask_tpc] = VOLUMES_IDS["tpc"]
            result["xe_density"][mask_tpc] = float(self.xenon_density_tpc)
            result["create_S2"][mask_tpc] = bool(self.create_S2_xenonnt_TPC)

        if np.any(mask_below_cathode):
            result["vol_id"][mask_below_cathode] = VOLUMES_IDS["below_cathode"]
            result["xe_density"][mask_below_cathode] = float(self.xenon_density_below_cathode)
            result["create_S2"][mask_below_cathode] = bool(self.create_S2_xenonnt_below_cathode)

        if np.any(mask_gas_phase):
            result["vol_id"][mask_gas_phase] = VOLUMES_IDS["gas_phase"]
            result["xe_density"][mask_gas_phase] = float(self.xenon_density_gas_phase)
            result["create_S2"][mask_gas_phase] = bool(self.create_S2_xenonnt_gas_phase)

        return result


@numba.njit
def in_cylinder(x, y, z, min_z, max_z, max_r):
    """Function which checks if a given set of coordinates is within the
    boundaries of the specified cylinder.

    Args:
        x,y,z: Coordinates of the interaction
        min_z: Inclusive lower z boundary
        max_z: Exclusive upper z boundary
        max_r: Exclusive radial boundary
    """
    r = np.sqrt(x**2 + y**2)
    m = r < max_r
    m = m & (z < max_z)
    m = m & (z >= min_z)
    return m
