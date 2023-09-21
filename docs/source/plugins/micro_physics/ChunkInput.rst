==========
ChunkInput
==========

Plugin Description
==================
Plugin to read XENONnT Geant4 root or csv files. The plugin can distribute the events
in time based on a source rate and will create multiple chunks of data if needed.
A detailed description of the dynamic chunking process is given in :doc:`/tech_features/DeterministicSeed`

Technical Details
-----------------

.. code-block:: python

   depends_on = ()
   provides = "geant4_interactions"
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
     - time of the energy deposit
   * - endtime
     - int64
     - endtime of the energy deposit (will be the same as time)
   * - x
     - float64
     - x position of the energy deposit
   * - y
     - float64
     - y position of the energy deposit
   * - z
     - float64
     - z position of the energy deposit
   * - t
     - float64
     - time of the energy deposit with respect to the start of the event
   * - ed
     - float32
     - Energy deposit in keV
   * - type
     - <U10
     - Particle type 
   * - trackid
     - int16
     - Geant4 track ID
   * - parenttype
     - <U10
     - Particle type of the parent particle
   * - parentid
     - int16
     - trackid of the parent particle
   * - creaproc
     - <U10
     - Geant4 process creating the particle
   * - edproc
     - <U10
     - Geant4 process destroying the particle
   * - evtid
     - int32
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
   * - path
     - 
     - False
     - Path to the file to simulate from excluding the file name
   * - file_name
     - 
     - False
     - Name of the file to simulate from
   * - separation_scale
     - 1e8
     - True
     - Separation scale for the dynamic chunking in ns
   * - source_rate
     - 1
     - True
     - Source rate used to generate event times. Use a value >0 to generate event times in fuse. Use source_rate = 0 to use event times from the input file (only for csv input)
   * - cut_delayed
     - 4e14
     - True
     - All interactions happening after this time (including the event time) will be cut.
   * - n_interactions_per_chunk
     - 1e5
     - True
     - Minimum number of interaction per chunk.
   * - entry_start
     - 0
     - True
     - Geant4 event to start simulation from. 
   * - entry_stop
     - None
     - True
     - Geant4 event to stop simulation at. If None, all events are simulated.
   * - cut_by_eventid
     - False
     - True
     - If selected, entry_start and entry_stop act on the G4 event id, and not the entry number (default)
   * - nr_only
     - False
     - True
     - Filter only nuclear recoil events (maximum ER energy deposit 10 keV)
   * - deterministic_seed
     - True
     - True
     - Set the random seed from lineage and run_id (True), or pull the seed from the OS (False).