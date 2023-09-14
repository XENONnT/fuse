=============
ElectronDrift
=============

Plugin Description
==================

Plugin to simulate the loss of electrons during the drift from the 
interaction site to the liquid gas interface. The plugin simulates the 
effect of a charge insensitive volume and the loss of electrons due to 
impurities. 

Technical Details
-----------------

.. code-block:: python

   depends_on = ("microphysics_summary")
   provides = "drifted_electrons"
   data_kind = "interactions_in_roi"

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
     - time of the energy deposit
   * - endtime
     - int64
     - endtime of the energy deposit (will be the same as time)
   * - n_electron_interface
     - int32
     - Number of electrons reaching the liquid gas interface
   * - drift_time_mean
     - int32
     - Mean drift time of the electrons in the cluster
   * - drift_time_spread
     - int32
     - Spread of the drift time of the electrons in the cluster
   * - x_obs
     - float32
     - observed x position of the cluster at liquid-gas interface
   * - y_obs
     - float32
     - observed y position of the cluster at liquid-gas interface
   * - z_obs
     - float32
     - observed z position of the cluster after field distortion correction. 


Config Options
==============

.. list-table::
   :widths: 25 25 10 40
   :header-rows: 1

   * - Option
     - default
     - track
     - comment
   * - debug
     - False
     - False
     - Show debug information during simulation
   * - drift_velocity_liquid
     - 
     - True
     - Drift velocity of electrons in the liquid xenon
   * - drift_time_gate
     - 
     - True
     - Electron drift time from the gate in ns
   * - diffusion_constant_longitudinal
     - 
     - True
     - Longitudinal electron drift diffusion constant
   * - electron_lifetime_liquid
     - 
     - True
     - Electron lifetime in liquid xenon
   * - enable_field_dependencies
     - 
     - True
     - Field dependencies during electron drift
   * - tpc_length
     - 
     - True
     - Length of the XENONnT TPC
   * - field_distortion_model
     - 
     - True
     - Model for the electric field distortion
   * - field_dependencies_map_tmp
     - 
     - True
     - Map for the electric field dependencies
   * - diffusion_longitudinal_map_tmp
     - 
     - True
     - Longitudinal diffusion map
   * - fdc_map_fuse
     - 
     - True
     - Field distortion map used in fuse (Check if we can remove _fuse from the name)