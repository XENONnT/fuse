======================
SecondaryScintillation
======================

Plugin Description
==================
Plugin to simulate the secondary scintillation process in the gas phase. 

Technical Details
-----------------

.. code-block:: python

   depends_on = ("drifted_electrons","extracted_electrons" ,"electron_time")
   provides = ("s2_photons", "s2_photons_sum")
   data_kind = {"s2_photons": "individual_electrons",
                "s2_photons_sum" : "interactions_in_roi"
                }


Provided Columns
================

s2_photons
----------

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Field Name
     - Data Type
     - Comment
   * - time
     - int64
     - Time of the individual electron reaching the gas phase
   * - endtime
     - int64
     - Endtime of the electron reaching the gas phase (same values as time)
   * - n_s2_photons
     - int64
     - number of photons produced by the extracted electron


s2_photons_sum
--------------

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
   * - sum_s2_photons
     - int64
     - sum of all photons produced by electrons originating from the same cluster


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
   * - electron_trapping_time
     - 
     - True
     - Time scale electrons are trapped at the liquid gas interface
   * - s2_gain_spread
     - 
     - True
     - Spread of the S2 gain
   * - s2_secondary_sc_gain
     - 
     - True
     - Secondary scintillation gain
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
   * - se_gain_from_map
     - 
     - True
     - Boolean indication if the secondary scintillation gain is taken from a map
   * - p_double_pe_emision
     - 
     - True
     - Probability of double photo-electron emission
   * - se_gain_map
     - 
     - True
     - Map of the single electron gain
   * - s2_correction_map
     - 
     - True
     - S2 correction map
   * - gains
     - 
     - True
     - PMT gains
   * - s2_pattern_map
     - 
     - True
     - S2 pattern map
   * - deterministic_seed
     - True
     - True
     - Set the random seed from lineage and run_id (True), or pull the seed from the OS (False).