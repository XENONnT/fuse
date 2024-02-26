==========
ChunkInput
==========

Link to source: `here <https://github.com/XENONnT/fuse/blob/main/fuse/plugins/micro_physics/input.py>`_.

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
     - Time of the energy deposit
   * - endtime
     - int64
     - Endtime of the energy deposit (will be the same as time)
   * - x
     - float64
     - x position of the energy deposit [cm]
   * - y
     - float64
     - y position of the energy deposit [cm]
   * - z
     - float64
     - z position of the energy deposit [cm]
   * - t
     - float64
     - Time with respect to the start of the event [ns]
   * - ed
     - float32
     - Energy deposit in keV
   * - type
     - <U18
     - Particle type 
   * - trackid
     - int16
     - Geant4 track ID
   * - parenttype
     - <U18
     - Particle type of the parent particle
   * - parentid
     - int16
     - Trackid of the parent particle
   * - creaproc
     - <U25
     - Geant4 process creating the particle
   * - edproc
     - <U25
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
   * - path
     - 
     - False
     - Path to the input file
   * - file_name
     - 
     - False
     - Name of the input file
   * - separation_scale
     - 1e8
     - True
     - Separation scale for the dynamic chunking in [ns]
   * - source_rate
     - 1
     - True
     - Source rate used to generate event times. Use a value >0 to generate event times in fuse. Use source_rate = 0 to use event times from the input file (only for csv input)
   * - cut_delayed
     - 9e18
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
