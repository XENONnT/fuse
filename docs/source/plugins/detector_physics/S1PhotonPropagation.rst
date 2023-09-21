===================
S1PhotonPropagation
===================

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
     - time of individual s1 photons
   * - endtime
     - int64
     - endtime of individual s1 photons (will be the same as time)
   * - channel
     - int16
     - PMT channel of the detected photon
   * - dpe
     - bool
     - Boolean indicating weather the photon will create a double photoelectron emisison or not
   * - photon_gain
     - int32
     - Gain of the PMT channel

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
   * - debug
     - False
     - False
     - Show debug information during simulation
   * - p_double_pe_emision
     - 
     - True
     - Probability of double photo-electron emission
   * - pmt_transit_time_spread
     - 
     - True
     - Spread of the PMT transit times
   * - pmt_transit_time_mean
     - 
     - True
     - Mean of the PMT transit times
   * - pmt_circuit_load_resistor
     - 
     - True
     - PMT circuit load resistor
   * - digitizer_bits
     - 
     - True
     - Number of bits of the digitizer boards
   * - digitizer_voltage_range
     - 
     - True
     - Voltage range of the digitizer boards
   * - n_top_pmts
     - 
     - True
     - Number of PMTs on top array
   * - n_tpc_pmts
     - 
     - True
     - Number of PMTs in the TPC
   * - gains
     - 
     - True
     - PMT gains
   * - photon_area_distribution
     - 
     - True
     - Photon area distribution
   * - s1_pattern_map
     - 
     - True
     - S1 pattern map
   * - deterministic_seed
     - True
     - True
     - Set the random seed from lineage and run_id (True), or pull the seed from the OS (False).

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
     - Maximum recombination time
   * - s1_optical_propagation_spline
     - 
     - False
     - Spline for the optical propagation