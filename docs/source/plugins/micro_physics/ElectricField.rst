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
     - time of the energy deposit
   * - endtime
     - int64
     - endtime of the energy deposit (will be the same as time)
   * - e_field
     - int64
     - Electric field value at the cluster position. 

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
   * - efield_map
     - 
     - True
     - electric field map