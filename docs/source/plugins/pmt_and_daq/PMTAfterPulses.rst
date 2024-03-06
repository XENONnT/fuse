==============
PMTAfterPulses
==============

Link to source: `here <https://github.com/XENONnT/fuse/blob/main/fuse/plugins/pmt_and_daq/pmt_afterpulses.py>`_.

Plugin Description
==================
Plugin to simulate PMT afterpulses using a precomputed afterpulse cumulative distribution function.
In the simulation afterpulses will be saved as a list of "pseudo" photons.
These "photons" can then be combined with real photons from S1 and S2 signals to create a waveform.

Technical Details
-----------------

.. code-block:: python

   depends_on = ("propagated_s2_photons", "propagated_s1_photons")
   provides = "pmt_afterpulses"
   data_kind = "AP_photons"
   __version__ = "0.3.0"

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
     - Time of afterpulse [ns]
   * - endtime
     - int64
     - Endtime of the afterpulse [ns] (same as time)
   * - channel
     - int16
     - PMT channel of the afterpulse
   * - dpe
     - bool
     - Afterpulse creates a double photo-electron emission (always False)
   * - photon_gain
     - int32
     - Sampled PMT gain for the afterpulse
   * - cluster_id
     - int32
     - ID of the cluster creating the photon
   * - photon_type
     - int8
     - Type of the photon. S1 (1), S2 (2) or PMT AP (0)

Config Options
==============

.. list-table::
   :widths: 25 25 10 40
   :header-rows: 1

   * - Option
     - default
     - track
     - comment
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
   * - gain_model_mc
     - 
     - True
     - PMT gain model
   * - photon_ap_cdfs
     - 
     - True
     - Afterpuse cumulative distribution functions