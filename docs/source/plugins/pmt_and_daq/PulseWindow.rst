===========
PulseWindow
===========

Plugin Description
==================
Plugin to compute time intervals (called `pulse_windows``) in which the 
PMT response of photons can overlap. Additionally a `pulse_id` is computed 
for each propagated photon to identify the pulse window it belongs to.

Technical Details
-----------------

.. code-block:: python

   depends_on = ("photon_summary")
   provides = ("pulse_windows", "pulse_ids")
   data_kind = {"pulse_windows": "pulse_windows",
                "pulse_ids" : "propagated_photons"
                }

Provided Columns
================

pulse_windows
-------------

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Field Name
     - Data Type
     - Comment
   * - time
     - int64
     - Time of the individual electron reaching the gas phase
   * - length
     - int32
     - Length of the interval in samples
   * - dt
     - int16
     - Width of one sample [ns]
   * - channel
     - int16
     - Channel/PMT number
   * - pulse_id
     - int64
     - ID of the pulse window


pulse_ids
---------

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
   * - pulse_id
     - int64
     - pulse id to map the photon to the pulse window it belongs to

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
   * - dt
     - 
     - True
     - Width of one sample [ns]
   * - samples_after_pulse_center
     - 
     - True
     - Number of samples after the pulse center
   * - samples_to_store_after
     - 
     - True
     - Number of samples to store after the pulse center
   * - samples_before_pulse_center
     - 
     - True
     - Number of samples before the pulse center
   * - samples_to_store_before
     - 
     - True
     - Number of samples to store before the pulse center
   * - n_tpc_pmts
     - 
     - True
     - Number of PMTs in the TPC