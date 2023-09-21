==============
PMTAfterPulses
==============

Plugin Description
==================
Plugin to simulate PMT afterpulses using a precomputed afterpulse cumulative distribution function.
In XENON simulations (WFSim and now fuse) afterpulses will be saved as a list of "pseudo" photons.
These "photons" can then be combined with real photons from S1 and S2 signals to create 
detectable signals.

Technical Details
-----------------

.. code-block:: python

   depends_on = ("propagated_s2_photons", "propagated_s1_photons")
   provides = "pmt_afterpulses"
   data_kind = "AP_photons"

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
   * - pmt_ap_t_modifier
     - 
     - True
     - PMT afterpulse time modifier
   * - pmt_ap_modifier
     - 
     - True
     - PMT afterpulse modifier
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
   * - gains
     - 
     - True
     - PMT gains
   * - photon_ap_cdfs
     - 
     - True
     - Afterpuse cumulative distribution functions
   * - deterministic_seed
     - True
     - True
     - Set the random seed from lineage and run_id (True), or pull the seed from the OS (False).