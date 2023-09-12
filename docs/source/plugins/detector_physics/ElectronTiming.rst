==============
ElectronTiming
==============

Plugin Description
==================
Plugin to simulate the arrival times of electrons extracted from the liquid phase. It includes both the 
drift time and the time needed for the extraction. 

Technical Details
-----------------

.. code-block:: python

   depends_on = ("drifted_electrons", "extracted_electrons")
   provides = "electron_time"
   data_kind = "individual_electrons"

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
     - Time of the individual electron reaching the gas phase
   * - endtime
     - int64
     - Endtime of the electron reaching the gas phase (same values as time)
   * - x
     - float64
     - x position of the electron
   * - y
     - float64
     - y position of the electron
   * - order_index
     - int64
     - index to order the electrons like they were initially produced. This way they can be easier mapped to the corresponding interactions_in_roi.

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
   * - deterministic_seed
     - True
     - True
     - Set the random seed from lineage and run_id (True), or pull the seed from the OS (False).