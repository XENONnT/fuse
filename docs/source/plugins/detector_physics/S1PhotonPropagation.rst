===================
S1PhotonPropagation
===================

Link to source: `here <https://github.com/XENONnT/fuse/blob/main/fuse/plugins/detector_physics/s1_photon_propagation.py>`_.

Plugin Description
==================
Plugin to simulate the propagation of S1 photons in the detector. Photons are 
randomly assigned to PMT channels based on their starting position and 
the timing of the photons is calculated.

The plugin is split into a `S1PhotonPropagationBase` class defining the compute
method as well as the photon channels calculation. The photon timing calculation
is implemented in the child plugin `S1PhotonPropagation` which inherits from
`S1PhotonPropagationBase`. This way we can add different timing calculations
without having to duplicate the photon channel calculation. 

Technical Details
-----------------

.. code-block:: python

   depends_on = ("s1_photons", "microphysics_summary")
   provides = "propagated_s1_photons"
   data_kind = "S1_photons"
   __version__ = "0.1.0"

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
     - Time of individual S1 photon [ns]
   * - endtime
     - int64
     - Endtime of individual S1 photon [ns] (same as time)
   * - channel
     - int16
     - PMT channel of the S1 photon
   * - dpe
     - bool
     - Photon creates a double photo-electron emission
   * - photon_gain
     - int32
     - Sampled PMT gain for the photon

Config Options
==============

S1PhotonPropagationBase plugin
-------------------------------

.. list-table::
   :widths: 25 25 10 40
   :header-rows: 1

   * - Option
     - default
     - track
     - comment
   * - p_double_pe_emision
     - 
     - True
     - Probability of double photo-electron emission
   * - pmt_transit_time_spread
     - 
     - True
     - Spread of the PMT transit times [ns]
   * - pmt_transit_time_mean
     - 
     - True
     - Mean of the PMT transit times [ns]
   * - pmt_circuit_load_resistor
     - 
     - True
     - PMT circuit load resistor [kg m^2/(s^3 A)] (PMT circuit resistance * electron charge * amplification factor * sampling frequency)
   * - digitizer_bits
     - 
     - True
     - Number of bits of the digitizer boards
   * - digitizer_voltage_range
     - 
     - True
     - Voltage range of the digitizer boards [V]
   * - n_top_pmts
     - 
     - True
     - Number of PMTs on top array
   * - n_tpc_pmts
     - 
     - True
     - Number of PMTs in the TPC
   * - gain_model_mc
     - 
     - True
     - PMT gain model
   * - photon_area_distribution
     - 
     - True
     - Photon area distribution
   * - s1_pattern_map
     - 
     - True
     - S1 pattern map

S1PhotonPropagation plugin
--------------------------

.. list-table::
   :widths: 25 25 10 40
   :header-rows: 1

   * - Option
     - default
     - track
     - comment
   * - maximum_recombination_time
     - 
     - False
     - Maximum recombination time [ns]
   * - s1_optical_propagation_spline
     - 
     - False
     - Spline for the optical propagation