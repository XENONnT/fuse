import strax
import straxen
import logging
import numba
import numpy as np

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.micro_physics.detector_volumes")


class VolumeSelection(strax.CutPlugin):
    """Plugin that evaluates if interactions are in a defined detector
    volume."""

    depends_on = "clustered_interactions"
    __version__ = "0.0.1"

    # Config options
    min_z = straxen.URLConfig(
        default= 0,
        type=(int, float),
        help="Lower limit of the volume [cm]",
    )

    max_z = straxen.URLConfig(
        type=(int, float),
        default= 0,
        help="Upper limit of the volume [cm]",
    )

    max_r = straxen.URLConfig(
        default= 0,
        type=(int, float),
        help="Radius of the Volume [cm]",
    )

    create_s2_in_volume = straxen.URLConfig(
        default= True,
        type=bool,
        help="Create S2s in the volume",
    )

    xenon_density_in_volume = straxen.URLConfig(
        default= 1,
        type=(int, float),
        help="Density of xenon in the volume",
    )

    def cut_by(self, clustered_interactions):

        mask = in_cylinder(
            clustered_interactions["x"],
            clustered_interactions["y"],
            clustered_interactions["z"],
            self.min_z,
            self.max_z,
            self.max_r,
        )

        return mask

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


class XENONnT_TPC(VolumeSelection):

    """Plugin to select only clusters in the XENONnT TPC."""
    child_plugin = True
    __version__ = "0.0.1"

    provides = "tpc_selection"
    cut_name = "tpc_selection"
    cut_description = "Selects events in the XENONnT TPC"

    xenonnt_z_cathode = straxen.URLConfig(
        default=-148.6515,  # Top of the cathode electrode
        track=True,
        type=(int, float),
        child_option=True,
        parent_option_name="min_z",
        help="z position of the XENONnT cathode [cm]",
    )

    xenonnt_z_gate_mesh = straxen.URLConfig(
        default=0.0,  # bottom of the gate electrode
        type=(int, float),
        child_option=True,
        parent_option_name="max_z",
        help="z position of the XENONnT gate mesh [cm]",
    )

    xenonnt_sensitive_volume_radius = straxen.URLConfig(
        default=66.4,
        type=(int, float),
        child_option=True,
        parent_option_name="max_r",
        help="Radius of the XENONnT TPC [cm]",
    )

    create_S2_xenonnt_tpc = straxen.URLConfig(
        default=True,
        type=bool,
        child_option=True,
        parent_option_name="create_s2_in_volume",
        help="Create S2s in the XENONnT TPC",
    )

    xenon_density_tpc = straxen.URLConfig(
        default=2.862,
        type=(int, float),
        child_option=True,
        parent_option_name="xenon_density_in_volume",
        help="Density of xenon in the TPC volume [g/cm3]",
    )

class XENONnT_BelowCathode(VolumeSelection):

    """Plugin to select only clusters  below the XENONnT cathode."""

    child_plugin = True
    __version__ = "0.0.1"

    provides = "below_cathode_selection"
    cut_name = "below_cathode_selection"
    cut_description = "Selects events below the XENONnT cathode."

    xenonnt_z_cathode = straxen.URLConfig(
        default=-148.6515,  # Top of the cathode electrode
        type=(int, float),
        child_option=True,
        parent_option_name="min_z",
        help="z position of the XENONnT cathode [cm]",
    )

    xenonnt_z_bottom_pmts = straxen.URLConfig(
        default=-154.6555,  # Top surface of the bottom PMT window
        type=(int, float),
        child_option=True,
        parent_option_name="max_z",
        help="z position of the XENONnT bottom PMT array [cm]",
    )

    xenonnt_sensitive_volume_radius = straxen.URLConfig(
        default=66.4,
        type=(int, float),
        child_option=True,
        parent_option_name="max_r",
        help="Radius of the XENONnT TPC [cm]",
    )

    create_S2_xenonnt_below_cathode = straxen.URLConfig(
        default=False,
        type=bool,
        child_option=True,
        parent_option_name="create_s2_in_volume",
        help="No S2s from below the cathode",
    )

    xenon_density_below_cathode = straxen.URLConfig(
        default=2.862,
        type=(int, float),
        child_option=True,
        parent_option_name="xenon_density_in_volume",
        help="Density of xenon in the below-cathode-volume [g/cm3]",
    )