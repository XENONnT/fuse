==============
ElectronTiming
==============

Link to source: `here <https://github.com/XENONnT/fuse/blob/main/fuse/plugins/detector_physics/electron_timing.py>`_.

Plugin Description
==================
Plugin to simulate the arrival times and positions of electrons extracted from the liquid phase. It includes both the 
drift time and the time needed for the extraction.

Technical Details
-----------------

.. code-block:: python

   depends_on = ("drifted_electrons", "extracted_electrons")
   provides = "electron_time"
   data_kind = "individual_electrons"
   __version__ = "0.1.1"

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
     - Time of the individual electron reaching the gas phase [ns]
   * - endtime
     - int64
     - Endtime of the electron reaching the gas phase [ns] (same values as time)
   * - x
     - float32
     - x position of the electron [cm]
   * - y
     - float32
     - y position of the electron [cm]
   * - order_index
     - int32
     - Index to order the electrons like they were initially produced. This way they can be easier mapped to the corresponding interactions_in_roi.

Config Options
==============

.. list-table::
   :widths: 25 25 10 40
   :header-rows: 1

   * - Option
     - default
     - track
     - comment
   * - electron_trapping_time
     - 
     - True
     - Time scale electrons are trapped at the liquid gas interface