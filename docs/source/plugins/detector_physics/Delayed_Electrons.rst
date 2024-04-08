========================
PhotoIonizationElectrons
========================

Link to source: `here <https://github.com/XENONnT/fuse/blob/main/fuse/plugins/detector_physics/delayed_electrons/photo_ionization_electrons.py>`_.

Plugin Description
==================

Plugin to simulate the emission of delayed electrons from photoionization in the liquid xenon using a
phenomenological model. The plugin uses the number of S2 photons per energy deposit as input and
creates delayed_interactions_in_roi. The simulation of delayed electrons can be enabled or disabled
using the config option enable_delayed_electrons. The amount of delayed electrons can be scaled using
the config option photoionization_modifier.

Technical Details
-----------------

.. code-block:: python

   depends_on = ("s2_photons_sum", "extracted_electrons", "s2_photons","electron_time", "microphysics_summary")
   provides = "photo_ionization_electrons"
   data_kind = "delayed_interactions_in_roi"
   __version__ = "0.0.2"

Provided Columns
================

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Field Name
     - Data Type
     - Comment
   * - time
     - int64
     - Time of the cluster [ns]
   * - endtime
     - int64
     - Endtime of the cluster [ns] (same as time)
   * - x
     - float32
     - x position of the cluster [cm]
   * - y
     - float32
     - y position of the cluster [cm]
   * - z
     - float32
     - z position of the cluster [cm]
   * - ed
     - float32
     - Energy of the cluster [keV]
   * - nestid
     - int8
     - NEST interaction type
   * - A
     - int8
     - Mass number of the interacting particle
   * - Z
     - int8
     - Charge number of the interacting particle
   * - evtid
     - int32
     - Geant4 event ID
   * - x_pri
     - float32
     - x position of the primary particle [cm]
   * - y_pri
     - float32
     - y position of the primary particle [cm]
   * - z_pri
     - float32
     - z position of the primary particle [cm]
   * - cluster_id
     - int32
     - ID of the cluster
   * - xe_density
     - float32
     - Xenon density at the cluster position.
   * - vol_id
     - int8
     - ID of the volume in which the cluster occured.
   * - create_S2
     - bool
     - Flag indicating if a cluster can create a S2 signal.
   * - e_field
     - float32
     - Electric field value at the cluster position [V/cm]
   * - photons
     - int32
     - Number of photons at interaction position.
   * - electrons
     - int32
     - Number of electrons at interaction position.
   * - excitons
     - int32
     - Number of excitons at interaction position.

Config Options
==============

.. list-table::
   :widths: 25 25 10 40
   :header-rows: 1

   * - Option
     - default
     - track
     - comment
   * - enable_delayed_electrons
     - False
     - True
     - Decide if you want to to enable delayed electrons from photoionization
   * - photoionization_time_cutoff
     -
     - True
     - Time window for photoionization after a S2 in [ns]
   * - photoionization_time_constant
     -
     - True
     - Timeconstant for photoionization in [ns]
   * - photoionization_modifier
     -
     - True
     - Photoionization modifier
   * - diffusion_constant_longitudinal
     -
     - True
     - Drift velocity of electrons in the liquid xenon
   * - drift_velocity_liquid
     -
     - True
     - Enable normalization of drif velocity map with drift_velocity_liquid
   * - drift_time_gate
     -
     - True
     - Electron drift time from the gate in ns
   * - tpc_radius
     -
     - True
     - Radius of the XENONnT TPC [cm]
   * - s2_secondary_sc_gain_mc
     -
     - True
     - Secondary scintillation gain
   * - p_double_pe_emision
     -
     - True
     - Probability of double photo-electron emission
   * - electron_extraction_yield
     -
     - True
     - Electron extraction yield
