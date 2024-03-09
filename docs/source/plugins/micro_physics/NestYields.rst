==========
NestYields
==========

Link to source: `here <https://github.com/XENONnT/fuse/blob/main/fuse/plugins/micro_physics/yields.py>`_.

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
   __version__ = "0.2.0"


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

No configuration options are available for this plugin.