===================
S2PhotonPropagation
===================

Link to source: `here <https://github.com/XENONnT/fuse/blob/main/fuse/plugins/detector_physics/s2_photon_propagation.py>`_.

Plugin Description
==================
Plugin to simulate the propagation of S2 photons in the detector. Photons are
randomly assigned to PMT channels based on their starting position and
the timing of the photons is calculated.

The plugin is split into a `S2PhotonPropagationBase` class defining the compute
method as well as the photon channels calculation. The photon timing calculation
is implemented in the child plugin `S2PhotonPropagation` which inherits from
`S2PhotonPropagationBase`. This way we can add different timing calculations
without having to duplicate the photon channel calculation. `S2PhotonPropagation` simulates
the photon timing using luminescence timing in the gas gap using a garfield simulation map, singlet and triplet decays and optical propagation.

Technical Details
-----------------

.. code-block:: python

   depends_on = ("electron_time","s2_photons", "extracted_electrons", "drifted_electrons", "s2_photons_sum", "microphysics_summary")
   provides = "propagated_s2_photons"
   data_kind = "S2_photons"
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
     - Time of individual S2 photon [ns]
   * - endtime
     - int64
     - Endtime of individual S2 photon [ns] (same as time)
   * - channel
     - int16
     - PMT channel of the S2 photon
   * - dpe
     - bool
     - Photon creates a double photo-electron emission
   * - photon_gain
     - int32
     - Sampled PMT gain for the photon
   * - cluster_id
     - int32
     - ID of the cluster creating the photon
   * - photon_type
     - int8
     - Type of the photon. S1 (1), S2 (2) or PMT AP (0)

Config Options
==============

S2PhotonPropagationBase plugin
-------------------------------

.. list-table::
   :widths: 25 25 10 40
   :header-rows: 1

   * - Option
     - default
     - track
     - comment
   * - p_double_pe_emision
     -
     - True
     - Probability of double photo-electron emission
   * - pmt_transit_time_spread
     -
     - True
     - Spread of the PMT transit times [ns]
   * - pmt_transit_time_mean
     -
     - True
     - Mean of the PMT transit times [ns]
   * - pmt_circuit_load_resistor
     -
     - True
     - PMT circuit load resistor [kg m^2/(s^3 A)] (PMT circuit resistance * electron charge * amplification factor * sampling frequency)
   * - digitizer_bits
     -
     - True
     - Number of bits of the digitizer boards
   * - digitizer_voltage_range
     -
     - True
     - Voltage range of the digitizer boards [V]
   * - n_top_pmts
     -
     - True
     - Number of PMTs on top array
   * - n_tpc_pmts
     -
     - True
     - Number of PMTs in the TPC
   * - gain_model_mc
     -
     - True
     - PMT gain model
   * - photon_area_distribution
     -
     - True
     - Photon area distribution
   * - phase_s2
     - "gas"
     - True
     - phase of the s2 producing region
   * - drift_velocity_liquid
     -
     - True
     - Drift velocity of electrons in the liquid xenon [cm/ns]
   * - tpc_length
     -
     - True
     - Length of the XENONnT TPC [cm]
   * - tpc_radius
     -
     - True
     - Radius of the XENONnT TPC [cm]
   * - diffusion_constant_transverse
     -
     - True
     - Transverse diffusion constant [cm^2/ns]
   * - s2_aft_skewness
     -
     - True
     - Skew of the S2 area fraction top
   * - s2_aft_sigma
     -
     - True
     - Width of the S2 area fraction top
  * - enable_diffusion_transverse_map
     -
     - True
     - Enable transverse diffusion map from field_dependencies_map_tmp
   * - s2_mean_area_fraction_top
     -
     - True
     - Mean S2 area fraction top
   * - s2_pattern_map
     -
     - True
     - S2 pattern map
   * - field_dependencies_map_tmp
     -
     - True
     - Map for the electric field dependencies
   * - singlet_fraction_gas
     -
     - True
     - Fraction of singlet states in GXe
   * - triplet_lifetime_gas
     -
     - True
     - Liftetime of triplet states in GXe [ns]
   * - singlet_lifetime_gas
     -
     - True
     - Liftetime of singlet states in GXe [ns]
   * - triplet_lifetime_liquid
     -
     - True
     - Liftetime of triplet states in LXe [ns]
   * - singlet_lifetime_liquid
     -
     - True
     - Liftetime of singlet states in LXe [ns]
   * - s2_secondary_sc_gain_mc
     -
     - True
     - Secondary scintillation gain [PE/e-]
   * - propagated_s2_photons_file_size_target
     - 300
     - False
     - Target for the propagated_s2_photons file size [MB]
   * - min_electron_gap_length_for_splitting
     - 1e5
     - False
     - Chunk can not be split if gap between photons is smaller than this value given in ns

S2PhotonPropagation plugin
--------------------------

.. list-table::
   :widths: 25 25 10 40
   :header-rows: 1

   * - Option
     - default
     - track
     - comment
   * - s2_luminescence_map
     -
     - False
     - Luminescence map for S2 Signals
   * - garfield_gas_gap_map
     -
     - False
     - Garfield gas gap map
   * - s2_optical_propagation_spline
     -
     - False
     - Spline for the optical propagation of S2 signals
