==========
NestYields
==========

Plugin Description
==================
Plugin that calculates the number of photons, electrons and excitons produced by
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
     - Time of the cluster [ns]
   * - endtime
     - int64
     - Endtime of the cluster [ns] (same as time)
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