=================
PMTResponseAndDAQ
=================

Link to source: `here <https://github.com/XENONnT/fuse/blob/main/fuse/plugins/pmt_and_daq/pmt_response_and_daq.py>`_.

Plugin Description
==================
Plugin to simulate the PMT response and DAQ effects. First the single PMT waveform
is simulated based on the photon timing and gain information. Next the waveform
is converted to ADC counts, noise and a baseline are added. Then hitfinding is performed
and the found intervals are split into multiple fragments of fixed length (if needed).
Finally the data is saved as `raw_records`.

Technical Details
-----------------

.. code-block:: python

   depends_on = ("photon_summary", "pulse_ids", "pulse_windows")
   provides = "raw_records"
   data_kind = "raw_records"
   __version__ = "0.1.3"

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
     - Start time since unix epoch [ns]
   * - length
     - int32
     - Length of the interval in samples
   * - dt
     - int16
     - Width of one sample [ns]
   * - channel
     - int16
     - Channel/PMT Number
   * - pulse_length
     - int32
     - Length of pulse to which the record belongs (without zero-padding)
   * - record_i
     - int16
     - Fragment number in the pulse
   * - baseline
     - int16
     - Baseline determined by the digitizer (if this is supported)
   * - data
     - int16, samples_per_record
     - Waveform data in raw ADC counts

Config Options
==============

.. list-table::
   :widths: 25 25 10 40
   :header-rows: 1

   * - Option
     - default
     - track
     - comment
   * - dt
     - 
     - True
     - Width of one sample [ns]
   * - pmt_circuit_load_resistor
     - 
     - True
     - PMT circuit load resistor [kg m^2/(s^3 A)] (PMT circuit resistance * electron charge * amplification factor * sampling frequency)
   * - external_amplification
     - 
     - True
     - External amplification factor
   * - digitizer_bits
     - 
     - True
     - Number of bits of the digitizer boards
   * - digitizer_voltage_range
     - 
     - True
     - Voltage range of the digitizer boards  [V]
   * - noise_data
     - 
     - True
     - Measured noise data
   * - pe_pulse_ts
     - 
     - True
     - Add a good description here
   * - pe_pulse_ys
     - 
     - True
     - Add a good description here
   * - pmt_pulse_time_rounding
     - 
     - True
     - Time rounding of the PMT pulse
   * - samples_after_pulse_center
     - 
     - True
     - Number of samples after the pulse center
   * - samples_before_pulse_center
     - 
     - True
     - Number of samples before the pulse center
   * - digitizer_reference_baseline
     - 
     - True
     - Digitizer reference baseline
   * - zle_threshold
     - 
     - True
     - Threshold for the zero length encoding
   * - trigger_window
     - 
     - True
     - Trigger window
   * - samples_to_store_before
     - 
     - True
     - Number of samples to store before the pulse center
   * - special_thresholds
     - 
     - True
     - Special thresholds for certain PMTs
   * - n_tpc_pmts
     - 
     - True
     - Number of PMTs in the TPC
   * - raw_records_file_size_target
     - 200
     - True
     - Target for the raw records file size [MB]
   * - min_records_gap_length_for_splitting
     - 1e5
     - True
     - Chunk can not be split if gap between pulses is smaller than this value given in ns