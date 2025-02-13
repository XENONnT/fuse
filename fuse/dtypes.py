import numpy as np


g4_fields = [
    (("Time with respect to the start of the event [ns]", "t"), np.float64),
    (("Energy deposit [keV]", "ed"), np.float32),
    (("Particle type", "type"), "<U18"),
    (("Geant4 track ID", "trackid"), np.int16),
    (("Particle type of the parent particle", "parenttype"), "<U18"),
    (("Trackid of the parent particle", "parentid"), np.int16),
    (("Geant4 process creating the particle", "creaproc"), "<U25"),
    (("Geant4 process responsible for the energy deposit", "edproc"), "<U25"),
    (("Geant4 event ID", "eventid"), np.int32),
]


primary_positions_fields = [
    (("x position of the primary particle [cm]", "x_pri"), np.float32),
    (("y position of the primary particle [cm]", "y_pri"), np.float32),
    (("z position of the primary particle [cm]", "z_pri"), np.float32),
]


deposit_positions_fields = [
    (("x position of the energy deposit [cm]", "x"), np.float32),
    (("y position of the energy deposit [cm]", "y"), np.float32),
    (("z position of the energy deposit [cm]", "z"), np.float32),
]


cluster_positions_fields = [
    (("x position of the cluster [cm]", "x"), np.float32),
    (("y position of the cluster [cm]", "y"), np.float32),
    (("z position of the cluster [cm]", "z"), np.float32),
]


cluster_id_fields = [
    (("Energy of the cluster [keV]", "ed"), np.float32),
    (("NEST interaction type", "nestid"), np.int8),
    (("ID of the cluster", "cluster_id"), np.int32),
]

csv_cluster_misc_fields = [
    (("Time of the interaction", "t"), np.int64),
    (("Geant4 event ID", "eventid"), np.int32),
]


cluster_misc_fields = [
    (("Mass number of the interacting particle", "A"), np.int16),
    (("Charge number of the interacting particle", "Z"), np.int16),
    (("Geant4 event ID", "eventid"), np.int32),
    (("Xenon density at the cluster position", "xe_density"), np.float32),
    (("ID of the volume in which the cluster occured", "vol_id"), np.int8),
    (("Flag indicating if a cluster can create a S2 signal", "create_S2"), np.bool_),
]


quanta_fields = [
    (("Number of photons at interaction position", "photons"), np.int32),
    (("Number of electrons at interaction position", "electrons"), np.int32),
    (("Number of excitons at interaction position", "excitons"), np.int32),
]


electric_fields = [
    (("Electric field value at the cluster position [V/cm]", "e_field"), np.float32),
]


propagated_photons_fields = [
    (("PMT channel of the photon", "channel"), np.int16),
    (("Photon creates a double photo-electron emission", "dpe"), np.bool_),
    (("Sampled PMT gain for the photon", "photon_gain"), np.int32),
    (("ID of the cluster creating the photon", "cluster_id"), np.int32),
    (("Type of the photon. S1 (1), S2 (2) or PMT AP (0)", "photon_type"), np.int8),
]
