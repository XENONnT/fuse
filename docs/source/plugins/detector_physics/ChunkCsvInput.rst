=============
ChunkCsvInput
=============

Link to source: `here <https://github.com/XENONnT/fuse/blob/main/fuse/plugins/detector_physics/csv_input.py>`_.

Plugin Description
==================
Plugin which reads a CSV file containing instructions for the detector physics simulation and returns the data in chunks.

Technical Details
-----------------

.. code-block:: python

   depends_on = ()
   provides = "microphysics_summary"
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
   * - x
     - float32
     - x position of the cluster [cm]
   * - y
     - float32
     - y position of the cluster [cm]
   * - z
     - float32
     - z position of the cluster [cm]
   * - photons
     - int32
     - Number of photons at interaction position.
   * - electrons
     - int32
     - Number of electrons at interaction position.
   * - excitons
     - int32
     - Number of excitons at interaction position.
   * - e_field
     - float32
     - Electric field value at the cluster position [V/cm]
   * - ed
     - float32
     - Energy of the cluster [keV]
   * - nestid
     - int8
     - NEST interaction type
   * - t
     - int64
     - Time of the interaction [ns]
   * - evtid
     - int32
     - Geant4 event ID

Config Options
==============

.. list-table::
   :widths: 25 25 10 40
   :header-rows: 1

   * - Option
     - default
     - track
     - comment
   * - input_file
     - 
     - False
     - CSV file to read
   * - separation_scale
     - 1e8
     - True
     - Separation scale for the dynamic chunking in [ns]
   * - source_rate
     - 1
     - True
     - Source rate used to generate event times. Use a value >0 to generate event times in fuse. Use source_rate = 0 to use event times from the input file (only for csv input)
   * - n_interactions_per_chunk
     - 1e5
     - True
     - Minimum number of interaction per chunk.