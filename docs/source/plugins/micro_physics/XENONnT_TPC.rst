============
XENONnT_TPC
============

Link to source: `here <https://github.com/XENONnT/fuse/blob/main/fuse/plugins/micro_physics/detector_volumes.py>`_.

Plugin Description
==================
Plugin to select only clusters in the XENONnT TPC. The TPC volume
is defined by the z position of the cathode and gate mesh and by the radius 
of the detector. For all clusters passing the volume selection `create_S2` is set
to `True`. 

Technical Details
-----------------

.. code-block:: python

   depends_on = ("clustered_interactions")
   provides = "tpc_interactions"
   data_kind = "tpc_interactions"
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
     - Xenon density at the cluster position.
   * - vol_id
     - int8
     - ID of the volume in which the cluster occured.
   * - create_S2
     - bool
     - Flag indicating if a cluster can create a S2 signal.


Config Options
==============

.. list-table::
   :widths: 25 25 10 40
   :header-rows: 1

   * - Option
     - default
     - track
     - comment
   * - xenonnt_z_cathode
     - -148.6515
     - True
     - z position of the XENONnT cathode [cm]
   * - xenonnt_z_gate_mesh
     - 0
     - True
     - z position of the XENONnT gate mesh [cm]
   * - xenonnt_sensitive_volume_radius
     - 66.4
     - True
     - Radius of the XENONnT TPC [cm]
   * - xenon_density_tpc
     - 2.862
     - True
     - Density of xenon in the TPC volume [g/cm3]
   * - create_S2_xenonnt_TPC
     - True
     - True
     - Create S2s in the XENONnT TPC