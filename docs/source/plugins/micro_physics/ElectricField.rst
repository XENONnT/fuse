=============
ElectricField
=============

Plugin Description
==================
Plugin that calculates the electric field values for the cluster position. 


Technical Details
-----------------

.. code-block:: python

   depends_on = ("interactions_in_roi")
   provides = "electric_field_values"
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
   * - e_field
     - int64
     - Electric field value at the cluster position [V/cm]

Config Options
==============

.. list-table::
   :widths: 25 25 10 40
   :header-rows: 1

   * - Option
     - default
     - track
     - comment
   * - efield_map
     - 
     - True
     - Map of the electric field in the detector