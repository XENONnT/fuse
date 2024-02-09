===========
FindCluster
===========

Link to source: `here <https://github.com/XENONnT/fuse/blob/main/fuse/plugins/micro_physics/find_cluster.py>`_.

Plugin Description
==================
Plugin to find clusters of energy deposits. This plugin is performing the first half 
of the microclustering process. Energy deposits are grouped into clusters based on
their proximity to each other in 3D space and time. The clustering is performed using
a 1D temporal clustering algorithm followed by 3D DBSCAN spacial clustering.

Technical Details
-----------------

.. code-block:: python

   depends_on = ("geant4_interactions")
   provides = "cluster_index"
   data_kind = "geant4_interactions"


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
     - Time of the energy deposit
   * - endtime
     - int64
     - Endtime of the energy deposit (will be the same as time)
   * - cluster_ids
     - int32
     - Cluster index of the energy deposit

Config Options
==============

.. list-table::
   :widths: 25 25 10 40
   :header-rows: 1

   * - Option
     - default
     - track
     - comment
   * - micro_separation_time
     - 10
     - True
     - Clustering time (ns)
   * - micro_separation
     - 0.005
     - True
     - DBSCAN clustering distance (mm)