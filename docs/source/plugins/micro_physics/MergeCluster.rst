============
MergeCluster
============

Plugin Description
==================
Plugin that merges energy deposits with the same cluster index into a single interaction. 
The 3D postiion is calculated as the energy weighted average of the 3D positions of the energy deposits.
The time of the merged cluster is calculated as the energy weighted average of the times of the energy deposits.
The energy of the merged cluster is the sum of the individual energy depositions. The cluster is then 
classified based on either the first interaction in the cluster or the most energetic interaction.

Technical Details
-----------------

.. code-block:: python

   depends_on = ("geant4_interactions", "cluster_index")
   provides = "clustered_interactions"
   data_kind = "clustered_interactions"

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
     - Time of the cluster
   * - endtime
     - int64
     - Endtime of the cluster (same as time)
   * - x
     - float32
     - x position of the cluster [cm]
   * - y
     - float32
     - y position of the cluster [cm]
   * - z
     - float32
     - z position of the cluster [cm]
   * - ed
     - float32
     - Energy of the cluster [keV]
   * - nestid
     - int8
     - NEST interaction type
   * - A
     - int8
     - Mass number of the interacting particle
   * - Z
     - int8
     - Charge number of the interacting particle
   * - evtid
     - int32
     - Geant4 event ID
   * - x_pri
     - float32
     - x position of the primary particle [cm]
   * - y_pri
     - float32
     - y position of the primary particle [cm]
   * - z_pri
     - float32
     - z position of the primary particle [cm]
   * - xe_density
     - float32
     - Xenon density at the cluster position. Will be set later.
   * - vol_id
     - int8
     - ID of the volume in which the cluster occured. Will be set later.
   * - create_S2
     - bool
     - Flag indicating if a cluster can create a S2 signal. Will be set later.


Config Options
==============

.. list-table::
   :widths: 25 25 10 40
   :header-rows: 1

   * - Option
     - default
     - track
     - comment
   * - tag_cluster_by
     - "energy"
     - True
     - Decide if you tag the cluster according to first interaction (time) or most energetic (energy) one.)