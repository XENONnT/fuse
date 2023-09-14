==========
NestYields
==========

Plugin Description
==================
Plugin that calculates the number of photons and electrons produced by
energy deposit using nestpy. 


Technical Details
-----------------

.. code-block:: python

   depends_on = ("interactions_in_roi", "electric_field_values")
   provides = "quanta"
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
   * - photons
     - int32
     - Number of photons at interaction position. 
   * - electrons
     - int32
     - Number of electrons at interaction position.
   * - excitons
     - int32
     - Number of excitons at interaction position. 


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
   * - deterministic_seed
     - True
     - True
     - Set the random seed from lineage and run_id (True), or pull the seed from the OS (False).