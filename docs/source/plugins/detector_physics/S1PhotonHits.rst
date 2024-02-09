============
S1PhotonHits
============

Link to source: `here <https://github.com/XENONnT/fuse/blob/main/fuse/plugins/detector_physics/s1_photon_hits.py>`_.

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
     - Time of the cluster [ns]
   * - endtime
     - int64
     - Endtime of the cluster [ns] (same as time)
   * - n_s1_photon_hits
     - int32
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
   * - pmt_circuit_load_resistor
     - 
     - True
     - PMT circuit load resistor [Unit!]
   * - digitizer_bits
     - 
     - True
     - Number of bits of the digitizer boards
   * - digitizer_voltage_range
     - 
     - True
     - Voltage range of the digitizer boards [Unit!]
   * - gain_model_mc
     - 
     - True
     - PMT gain model
   * - s1_pattern_map
     - 
     - True
     - S1 pattern map
   * - p_double_pe_emision
     - 
     - True
     - Probability of double photo-electron emission
   * - s1_detection_efficiency
     - 
     - True
     - S1 detection efficiency