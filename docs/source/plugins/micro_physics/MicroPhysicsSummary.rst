===================
MicroPhysicsSummary
===================

Link to source: `here <https://github.com/XENONnT/fuse/blob/main/fuse/plugins/micro_physics/microphysics_summary.py>`_.

Plugin Description
==================
MergeOnlyPlugin that summarizes the fuse microphysics simulation results into a single output. 

Technical Details
-----------------

.. code-block:: python

   depends_on = ("interactions_in_roi", "quanta", "electric_field_values")
   provides = "microphysics_summary"
   __version__ = "0.1.0"
