====================
XENONnT_BelowCathode
====================

Link to source: `here <https://github.com/XENONnT/fuse/blob/main/fuse/plugins/micro_physics/detector_volumes.py>`_.

Plugin Description
==================
Plugin to select only clusters  below the XENONnT cathode. The volume
is defined by the z position of the cathode and bottom PMTs and by the radius
of the detector. For all clusters passing the volume selection `create_S2` is set
to `False`.

Technical Details
-----------------

.. code-block:: python

   depends_on = ("clustered_interactions")
   provides = "below_cathode_interactions"
   data_kind = "below_cathode_interactions"
   __version__ = "0.3.0"

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
   * - cluster_id
     - int32
     - ID of the cluster
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
   * - xenonnt_z_bottom_pmts
     - -154.6555
     - True
     - z position of the XENONnT bottom PMT array [cm]
   * - xenonnt_sensitive_volume_radius
     - 66.4
     - True
     - Radius of the XENONnT TPC [cm]
   * - xenon_density_below_cathode
     - 2.862
     - True
     - Density of xenon in the below-cathode-volume [g/cm3]
   * - create_S2_xenonnt_below_cathode
     - False
     - True
     - No S2s from below the cathode
