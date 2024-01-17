=============
VolumesMerger
=============

Plugin Description
==================
fuse VerticalMergerPlugin that concatenates the clusters that are in the XENONnT TPC or the volume below the cathode.

Technical Details
-----------------

.. code-block:: python

   depends_on = ("tpc_interactions", "below_cathode_interactions")
   provides = "interactions_in_roi"
   data_kind = "interactions_in_roi"