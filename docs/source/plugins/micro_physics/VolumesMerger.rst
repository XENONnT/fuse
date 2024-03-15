=============
VolumesMerger
=============

Link to source: `here <https://github.com/XENONnT/fuse/blob/main/fuse/plugins/micro_physics/detector_volumes.py>`_.

Plugin Description
==================
Plugin that concatenates the clusters that are in the XENONnT TPC or the volume below the cathode.

Technical Details
-----------------

.. code-block:: python

   depends_on = ("tpc_interactions", "below_cathode_interactions")
   provides = "interactions_in_roi"
   data_kind = "interactions_in_roi"
   __version__ = "0.1.0"
