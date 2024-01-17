================
Dynamic Chunking
================

In strax all data is processed in chunks. Chunks are time intervals containing some data.
As fuse is build on the strax framework we need to follow the `basic rules of chunking <https://strax.readthedocs.io/en/latest/advanced/chunking.html>`_. 
The main takeaways for fuse are that chunks are independent and continous in time. 


Input Chunking
==============

Input data (from Geant4 or csv) comes as one singe file that must be split into chunks.
fuse has dedicated input plugins to do this. fuse will split the input data based on the
interaction times. fuse searches for time gaps in between the interaction that are large enough
to accomodate the physics processes like e.g. the electron drift. The input chunking in fuse
can be controlled by these arguments:

- `separation_scale`: minimum time gap between interactions in ns to qualify as a split time
- `n_interactions_per_chunk`: minimum number of interactions per chunk

Memory intensive plugins
========================
In general fuse moves from leightweigt input data (e.g. list of energy deposits) to more heavy data
like e.g. long list of individual photons. To avoid memory issuse we make use of a new strax feature
to split incoming chunks into smaller chunks based on the estimated output memory size.
This is done in the S2PhotonPropagation and PMTResponseAndDAQ plugins and can be controlled by these arguments:

- `propagated_s2_photons_file_size_target`: upper limit for the propagated_s2_photons file size in MB
- `min_electron_gap_length_for_splitting`: minimum gap length in ns between electrons
- `raw_records_file_size_target`: upper limit for the raw_records file size in MB
- `min_records_gap_length_for_splitting`: minimum gap length in ns between two photon pulses