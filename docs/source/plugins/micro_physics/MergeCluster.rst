============
MergeCluster
============

Plugin Description
==================
This plugin merges energy deposits with the same cluster ID into a single interaction. 
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
     - time of cluster
   * - endtime
     - int64
     - endtime of the cluster (will be the same as time)
   * - x
     - float32
     - x position of the cluster
   * - y
     - float32
     - y position of the cluster
   * - z
     - float32
     - z position of the cluster
   * - ed
     - float64
     - Energy of the cluster in keV
   * - nestid
     - int64
     - NEST interaction type 
   * - A
     - int64
     - Mass number of the interacting particle
   * - Z
     - int64
     - Charge number of the interacting particle
   * - evtid
     - int64
     - Geant4 event ID
   * - x_pri
     - float32
     - x position of the primary particle
   * - y_pri
     - float32
     - y position of the primary particle
   * - z_pri
     - float32
     - z position of the primary particle
   * - xe_density
     - float32
     - Xenon density at cluster position. Will be set in a later plugin. 
   * - vol_id
     - int64
     - ID of the volume in which the cluster occured. Will be set in a later plugin.
   * - create_S2
     - bool8
     - Flag indicating if a cluster can create a S2 signal (True) or not (False)


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
   * - tag_cluster_by
     - "energy"
     - True
     - Decide if you tag the cluster (particle type, energy depositing process) according to first interaction in it (time) or most energetic (energy))