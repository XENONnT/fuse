import strax
import numba
import numpy as np

from .plugin import FuseBasePlugin


class VolumePlugin(FuseBasePlugin):
    """Plugin that evaluates if interactions are in a defined detector
    volume."""

    save_when = strax.SaveWhen.TARGET

    def setup(self):
        super().setup()

    # Forbid rechunking
    rechunk_on_save = False

    __version__ = "0.0.1"

    def in_ROI(self, interactions, min_z, max_z, max_r):
        """Function that evaluates if an interaction is in the ROI."""
        mask = in_cylinder(
            interactions["x"], interactions["y"], interactions["z"], min_z, max_z, max_r
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
