================
Simulation Chain
================

In fuse the simulation steps are put into separate plugins that are chained together.
This makes it easy to exchange or add new simulation steps while keeping the rest of the simulation intact.
As a bonus, the output of the simulation steps can be easily accessed.

Simulation overview
===================

.. image:: figures/fuse_simulation_overview.pdf
    :alt: Overview of the simulation chain.

The simulation steps can be divided into three main steps. These steps are: 

1. :doc:`Microphysics Simulation <microphysics_simulation>` - In this step the microphysics response of the
   xenon in the detector is simulated. This includes the clustering of energy deposits and the calculation of the
   number of photons and electrons.
2. :doc:`Detector Physics <detector_physics>`: In this section the effects of the detector on the photons and electrons are simulated. 
   Electrons are drifted from the interaction site to the liquid-gas interface where they are
   extraxted. The secondary scintillation process is simulated and the photons of the s1 and s2 signals are 
   propagated to the PMTs. Additionaly delayed electrons from photo ionization can be simulated.
3. :doc:`PMT and DAQ <pmt_and_daq>`: The last section of the simulation chain covers the simulation of the PMT response and the
   digitization of the PMT signals into XENONnT raw_records data. PMT afterpulses are simulated here as well.

A visualization of all plugins and their dependencies is shown in the figure below. 

.. image:: figures/fuse_simulation_chain.pdf
    :alt: Overview of the simulation chain.