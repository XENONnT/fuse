import strax
import numpy as np
import straxen
import logging
from ...volume_plugin import *
from ...vertical_merger_plugin import *

from ...common import FUSE_PLUGIN_TIMEOUT

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.micro_physics.detector_volumes')

class VolumesMerger(VerticalMergerPlugin):
    """
    Plugin that concatenates the clusters that are in the 
    XENONnT TPC or the volume below the cathode.
    """

    depends_on = ("tpc_interactions", "below_cathode_interactions")

    provides = "interactions_in_roi"
    data_kind = "interactions_in_roi"
    __version__ = "0.1.0"

    #Forbid rechunking
    rechunk_on_save = False

    input_timeout = FUSE_PLUGIN_TIMEOUT

    def compute(self, **kwargs):
        return super().compute(**kwargs)


# Fixed detector dimensions of XENONnT:
# See also: https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:analysis:coordinate_system
    
class XENONnT_TPC(VolumePlugin):
    """
    Plugin that evaluates if interactions are in the XENONnT TPC.
    Only these interactions are returned. The output of this plugin is 
    meant to go into a vertical merger plugin.
    """

    depends_on = ("clustered_interactions")

    provides = "tpc_interactions"
    data_kind = "tpc_interactions"
    __version__ = "0.1.1"

    #Can we import this from MergeCluster and just add the needed fields?
    dtype = [('x', np.float32),
             ('y', np.float32),
             ('z', np.float32),
             ('ed', np.float32),
             ('nestid', np.int8),
             ('A', np.int8),
             ('Z', np.int8),
             ('evtid', np.int32),
             ('x_pri', np.float32),
             ('y_pri', np.float32),
             ('z_pri', np.float32),
             ('xe_density', np.float32),
             ('vol_id', np.int8), 
             ('create_S2', np.bool_), 
            ]
    
    dtype = dtype + strax.time_fields

    #Config options
    #Define the TPC volume
    xenonnt_z_cathode = straxen.URLConfig(
        default = -148.6515,  # cm ... top of the cathode electrode
        type=(int, float),
        help='z position of the XENONnT cathode',
    )

    xenonnt_z_gate_mesh = straxen.URLConfig(
        default = 0.,  # bottom of the gate electrode
        type=(int, float),
        help='z position of the XENONnT gate mesh',
    )

    xenonnt_sensitive_volume_radius = straxen.URLConfig(
        default = 66.4,  # cm
        type=(int, float),
        help='Radius of the XENONnT TPC',
    )

    xenon_density_tpc = straxen.URLConfig(
        default = 2.862,
        type=(int, float),
        help='Density of xenon in the TPC volume in g/cm3',
    )

    create_S2_xenonnt_TPC = straxen.URLConfig(
        default=True, type=bool,
        help='Create S2s in the XENONnT TPC',
    )

    def compute(self, clustered_interactions):
        
        #Call the ROI function:
        mask = self.in_ROI(clustered_interactions,
                           self.xenonnt_z_cathode,
                           self.xenonnt_z_gate_mesh, 
                           self.xenonnt_sensitive_volume_radius
                           )
        
        tpc_interactions = clustered_interactions[mask]
        tpc_interactions["xe_density"] = self.xenon_density_tpc
        tpc_interactions["vol_id"] = 1 #Do we need this? -> Now just copied from epix
        tpc_interactions["create_S2"] = self.create_S2_xenonnt_TPC

        return tpc_interactions
    


class XENONnT_BelowCathode(VolumePlugin):
    """
    Plugin that evaluates if interactions are below the Cathode of XENONnT.
    Only these interactions are returned. The output of this plugin is 
    meant to go into a vertical merger plugin.
    """

    depends_on = ("clustered_interactions")

    provides = "below_cathode_interactions"
    data_kind = "below_cathode_interactions"
    __version__ = "0.1.1"

    #Can we import this from MergeCluster and just add the needed fields?
    dtype = [('x', np.float32),
             ('y', np.float32),
             ('z', np.float32),
             ('ed', np.float32),
             ('nestid', np.int8),
             ('A', np.int8),
             ('Z', np.int8),
             ('evtid', np.int32),
             ('x_pri', np.float32),
             ('y_pri', np.float32),
             ('z_pri', np.float32),
             ('xe_density', np.float32),
             ('vol_id', np.int8), 
             ('create_S2', np.bool_), 
            ]
    
    dtype = dtype + strax.time_fields

    #Config options
    #Define the volume
    xenonnt_z_cathode = straxen.URLConfig(
        default = -148.6515,  # cm ... top of the cathode electrode
        type=(int, float),
        help='xenonnt_z_cathode',
    )

    xenonnt_z_bottom_pmts = straxen.URLConfig(
        default = -154.6555,  # cm ... top surface of the bottom PMT window
        type=(int, float),
        help='z position of the XENONnT bottom PMT array',
    )

    xenonnt_sensitive_volume_radius = straxen.URLConfig(
        default = 66.4,  # cm
        type=(int, float),
        help='xenonnt_sensitive_volume_radius',
    )

    xenon_density_below_cathode = straxen.URLConfig(
        default = 2.862,
        type=(int, float),
        help='Density of xenon in the below-cathode-volume in g/cm3',
    )

    create_S2_xenonnt_below_cathode = straxen.URLConfig(
        default=False, type=bool,
        help='No S2s from below the cathode',
    )

    def compute(self, clustered_interactions):
        
        #Call the ROI function:
        mask = self.in_ROI(clustered_interactions,
                           self.xenonnt_z_bottom_pmts,
                           self.xenonnt_z_cathode, 
                           self.xenonnt_sensitive_volume_radius
                           )
        
        tpc_interactions = clustered_interactions[mask]
        tpc_interactions["xe_density"] = self.xenon_density_below_cathode
        tpc_interactions["vol_id"] = 2 #Do we need this? -> Now just copied from epix
        tpc_interactions["create_S2"] = self.create_S2_xenonnt_below_cathode

        return tpc_interactions
    


class XENONnT_GasPhase(VolumePlugin):
    """
    Plugin that evaluates if interactions are in the gas phase of XENONnT.
    Only these interactions are returned. The output of this plugin is 
    meant to go into a vertical merger plugin.
    """

    depends_on = ("clustered_interactions")

    provides = "gas_phase_interactions"
    data_kind = "gas_phase_interactions"
    __version__ = "0.1.0"

    #Can we import this from MergeCluster and just add the needed fields?
    dtype = [('x', np.float32),
             ('y', np.float32),
             ('z', np.float32),
             ('ed', np.float64),
             ('nestid', np.int64),
             ('A', np.int64),
             ('Z', np.int64),
             ('evtid', np.int64),
             ('x_pri', np.float32),
             ('y_pri', np.float32),
             ('z_pri', np.float32),
             ('xe_density', np.float32),
             ('vol_id', np.int64),
             ('create_S2', np.bool_),
            ]
    
    dtype = dtype + strax.time_fields

    #Config options
    #Define the volume
    xenonnt_z_top_pmts = straxen.URLConfig(
        default = 7.3936,  # cm
        type=(int, float),
        help='xenonnt_z_top_pmts',
    )

    xenonnt_z_lxe = straxen.URLConfig(
        default = 0.416, # cm ... liquid-gas interface
        type=(int, float),
        help='xenonnt_z_lxe',
    )

    xenonnt_sensitive_volume_radius = straxen.URLConfig(
        default = 66.4,  # cm
        type=(int, float),
        help='xenonnt_sensitive_volume_radius',
    )

    xenon_density_gas_phase = straxen.URLConfig(
        default = 0.0177,
        type=(int, float),
        help='xenon_density',
    )

    create_S2_xenonnt_gas_phase = straxen.URLConfig(
        default=False, type=bool,
        help='No S2s in gas',
    )

    def compute(self, clustered_interactions):
        
        #Call the ROI function:
        mask = self.in_ROI(clustered_interactions,
                           self.xenonnt_z_lxe,
                           self.xenonnt_z_top_pmts, 
                           self.xenonnt_sensitive_volume_radius
                           )
        
        tpc_interactions = clustered_interactions[mask]
        tpc_interactions["xe_density"] = self.xenon_density_gas_phase
        tpc_interactions["vol_id"] = 3 #Do we need this? -> Now just copied from epix
        tpc_interactions["create_S2"] = self.create_S2_xenonnt_gas_phase

        return tpc_interactions
    
