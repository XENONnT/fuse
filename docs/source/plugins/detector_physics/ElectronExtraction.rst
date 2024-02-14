==================
ElectronExtraction
==================

Link to source: `here <https://github.com/XENONnT/fuse/blob/main/fuse/plugins/detector_physics/electron_extraction.py>`_.

Plugin Description
==================
Plugin to simulate the loss of electrons during the extraction of drifted 
electrons from the liquid into the gas phase.

Technical Details
-----------------

.. code-block:: python

   depends_on = ("microphysics_summary", "drifted_electrons")
   provides = "extracted_electrons"
   data_kind = "interactions_in_roi"
   __version__ = "0.1.3"

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
   * - n_electron_extracted
     - int32
     - Number of electrons extracted into the gas phase


Config Options
==============

.. list-table::
   :widths: 25 25 10 40
   :header-rows: 1

   * - Option
     - default
     - track
     - comment
   * - s2_secondary_sc_gain_mc
     - 
     - True
     - Secondary scintillation gain [PE/e-]
   * - g2_mean
     - 
     - True
     - Mean value of the g2 gain [PE/e-]
   * - electron_extraction_yield
     - 
     - True
     - Electron extraction yield [electron_extracted/electron]
   * - ext_eff_from_map
     - 
     - True
     - Boolean indication if the extraction efficiency is taken from a map
   * - se_gain_from_map
     - 
     - True
     - Boolean indication if the secondary scintillation gain is taken from a map
   * - s2_correction_map
     - 
     - True
     - S2 correction map
   * - se_gain_map
     - 
     - True
     - Map of the single electron gain
   * - n_top_pmts
     - 
     - True
     - Number of PMTs on top array
   * - n_tpc_pmts
     - 
     - True
     - Number of PMTs in the TPC