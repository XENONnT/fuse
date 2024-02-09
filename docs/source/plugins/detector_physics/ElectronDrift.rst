=============
ElectronDrift
=============

Plugin Description
==================

Plugin to simulate the drift of electrons from the 
interaction site to the liquid gas interface. The plugin simulates the 
effect of a charge insensitive volume and the loss of electrons due to 
impurities. Additionally, the drift time and observed position is calculated.

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
     - Time of the cluster [ns]
   * - endtime
     - int64
     - Endtime of the cluster [ns] (same as time)
   * - n_electron_interface
     - int32
     - Number of electrons reaching the liquid gas interface
   * - drift_time_mean
     - int32
     - Mean drift time of the electrons in the cluster [ns]
   * - drift_time_spread
     - int32
     - Spread of the drift time of the electrons in the cluster [ns]
   * - x_obs
     - float32
     - Observed x position of the cluster at liquid-gas interface [cm]
   * - y_obs
     - float32
     - Observed y position of the cluster at liquid-gas interface [cm]
   * - z_obs
     - float32
     - Observed z position of the cluster after field distortion correction [cm]


Config Options
==============

.. list-table::
   :widths: 25 25 10 40
   :header-rows: 1

   * - Option
     - default
     - track
     - comment
   * - drift_velocity_liquid
     - 
     - True
     - Drift velocity of electrons in the liquid xenon [unit!]
   * - drift_time_gate
     - 
     - True
     - Electron drift time from the gate [ns]
   * - diffusion_constant_longitudinal
     - 
     - True
     - Longitudinal electron drift diffusion constant [unit!]
   * - electron_lifetime_liquid
     - 
     - True
     - Electron lifetime in liquid xenon [unit!]
   * - enable_field_dependencies
     - 
     - True
     - Field dependencies during electron drift 
   * - tpc_length
     - 
     - True
     - Length of the XENONnT TPC [cm]
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