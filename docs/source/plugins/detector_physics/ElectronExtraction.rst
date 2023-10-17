==================
ElectronExtraction
==================

Plugin Description
==================
Plugin to simulate the loss of electrons during the extraction of drifted 
electrons from the liquid into the gas phase. 

Technical Details
-----------------

.. code-block:: python

   depends_on = ("microphysics_summary", "drifted_electrons")
   provides = "extracted_electrons"
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
   * - n_electron_extracted
     - int32
     - Number of electrons extracted into the gas phase


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
   * - digitizer_voltage_range
     - 
     - True
     - Voltage range of the digitizer boards
   * - digitizer_bits
     - 
     - True
     - Number of bits of the digitizer boards
   * - pmt_circuit_load_resistor
     - 
     - True
     - PMT circuit load resistor 
   * - s2_secondary_sc_gain
     - 
     - True
     - Secondary scintillation gain
   * - g2_mean
     - 
     - True
     - mean value of the g2 gain. 
   * - electron_extraction_yield
     - 
     - True
     - Electron extraction yield
   * - ext_eff_from_map
     - 
     - True
     - Boolean indication if the extraction efficiency is taken from a map
   * - se_gain_from_map
     - 
     - True
     - Boolean indication if the secondary scintillation gain is taken from a map
   * - gain_model_mc
     - 
     - True
     - PMT gain model
   * - s2_correction_map
     - 
     - True
     - S2 correction map
   * - se_gain_map
     - 
     - True
     - Map of the single electron gain
   * - s2_pattern_map
     - 
     - True
     - S2 pattern map 
   * - deterministic_seed
     - True
     - True
     - Set the random seed from lineage and run_id (True), or pull the seed from the OS (False).