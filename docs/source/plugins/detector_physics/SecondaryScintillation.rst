======================
SecondaryScintillation
======================

Link to source: `here <https://github.com/XENONnT/fuse/blob/main/fuse/plugins/detector_physics/secondary_scintillation.py>`_.

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
   __version__ = "0.2.0"


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
     - Time of the individual electron reaching the gas phase [ns]
   * - endtime
     - int64
     - Endtime of the electron reaching the gas phase [ns] (same values as time)
   * - n_s2_photons
     - int32
     - Number of photons produced by the extracted electron


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
     - Time of the cluster [ns]
   * - endtime
     - int64
     - Endtime of the cluster [ns] (same as time)
   * - sum_s2_photons
     - int32
     - Sum of all photons produced by electrons originating from the same cluster


Config Options
==============

.. list-table::
   :widths: 25 25 10 40
   :header-rows: 1

   * - Option
     - default
     - track
     - comment
   * - s2_secondary_sc_gain_mc
     - 
     - True
     - Secondary scintillation gain [PE/e-]
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
   * - gain_model_mc
     - 
     - True
     - PMT gain model
   * - n_top_pmts
     - 
     - True
     - Number of PMTs on top array
   * - n_tpc_pmts
     - 
     - True
     - Number of PMTs in the TPC
   * - s2_mean_area_fraction_top
     - 
     - True
     - Mean S2 area fraction top
   * - s2_pattern_map
     - 
     - True
     - S2 pattern map