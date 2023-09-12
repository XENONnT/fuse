============
XENONnT_TPC
============

Plugin Description
==================
fuse volume plugin to select only clusters in the XENONnT TPC. The TPC volume
is defined by the z position of the cathode and gate mesh and by the radius 
of the detector. For all clusters passing the volume selection `create_S2` is set
to `True`. 

Technical Details
-----------------

.. code-block:: python

   depends_on = ("clustered_interactions")
   provides = "tpc_interactions"
   data_kind = "tpc_interactions"

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
     - Xenon density at cluster position.
   * - vol_id
     - int64
     - ID of the volume in which the cluster occured.
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
   * - xenonnt_z_cathode
     - -148.6515
     - True
     - z position of the XENONnT cathode
   * - xenonnt_z_gate_mesh
     - 0
     - True
     - z position of the XENONnT gate mesh
   * - xenonnt_sensitive_volume_radius
     - 66.4
     - True
     - Radius of the XENONnT TPC
   * - xenon_density_tpc
     - 2.862
     - True
     - Density of xenon in the TPC volume in g/cm3
   * - create_S2_xenonnt_TPC
     - True
     - True
     - Create S2s in the XENONnT TPC