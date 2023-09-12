============
S1PhotonHits
============

Plugin Description
==================
Plugin to simulate the number of detected S1 photons using a S1 light collection efficiency map. 

Technical Details
-----------------

.. code-block:: python

   depends_on = ("microphysics_summary")
   provides = "s1_photons"
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
   * - n_s1_photon_hits
     - int64
     - Number detected S1 photons

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
   * - s1_lce_correction_map
     - 
     - True
     - S1 light collection efficiency map
   * - p_double_pe_emision
     - 
     - True
     - Probability of double photo-electron emission
   * - s1_detection_efficiency
     - 
     - True
     - S1 detection efficiency
   * - deterministic_seed
     - True
     - True
     - Set the random seed from lineage and run_id (True), or pull the seed from the OS (False).