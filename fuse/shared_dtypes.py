import strax
import numpy as np

clustered_interactions_dtype = [
    (("x position of the cluster [cm]", "x"), np.float32),
    (("y position of the cluster [cm]", "y"), np.float32),
    (("z position of the cluster [cm]", "z"), np.float32),
    (("Energy of the cluster [keV]", "ed"), np.float32),
    (("NEST interaction type", "nestid"), np.int8),
    (("Mass number of the interacting particle", "A"), np.int16),
    (("Charge number of the interacting particle", "Z"), np.int16),
    (("Geant4 event ID", "evtid"), np.int32),
    (("x position of the primary particle [cm]", "x_pri"), np.float32),
    (("y position of the primary particle [cm]", "y_pri"), np.float32),
    (("z position of the primary particle [cm]", "z_pri"), np.float32),
    (("ID of the cluster", "cluster_id"), np.int32),
    (("Xenon density at the cluster position.", "xe_density"), np.float32),
    (("ID of the volume in which the cluster occured.", "vol_id"), np.int8),
    (("Flag indicating if a cluster can create a S2 signal.", "create_S2"), np.bool_),
]
clustered_interactions_dtype += strax.time_fields
