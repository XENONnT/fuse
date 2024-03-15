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

    #Forbid rechunking
    rechunk_on_save = False

    __version__ = "0.0.1"

    def infer_dtype(self):
        dtype = [(("x position of the cluster [cm]", "x"), np.float32),
                (("y position of the cluster [cm]", "y"), np.float32),
                (("z position of the cluster [cm]", "z"), np.float32),
                (("Energy of the cluster [keV]", "ed"), np.float32),
                (("NEST interaction type", "nestid"), np.int8),
                (("Mass number of the interacting particle", "A"), np.int16),
                (("Charge number of the interacting particle", "Z"), np.int16),
                (("Geant4 event ID", "evtid"), np.int32),
                #(("x position of the primary particle [cm]", "x_pri"), np.float32),
                #(("y position of the primary particle [cm]", "y_pri"), np.float32),
                #(("z position of the primary particle [cm]", "z_pri"), np.float32),
                (("ID of the cluster", "cluster_id"), np.int32),
                (("Xenon density at the cluster position.", "xe_density"), np.float32), 
                (("ID of the volume in which the cluster occured.", "vol_id"), np.int8),
                (("Flag indicating if a cluster can create a S2 signal.", "create_S2"), np.bool_),
                ]
    
        dtype = dtype + strax.time_fields
        
        return dtype

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
