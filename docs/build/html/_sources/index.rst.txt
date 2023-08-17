.. XENON fuse documentation master file, created by
   sphinx-quickstart on Wed Aug 16 16:06:00 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation of XENON fuse!
===========================================

**fuse** is the next generation XENONnT simulation software. It is a refactoring
of the XENONnT software `epix <https://github.com/XENONnT/epix>`_ and
`WFSim <https://github.com/XENONnT/WFSim>`_. fuse is build in a modular structure using
the `strax <https://github.com/XENONnT/straxen>`_ and `straxen <https://github.com/AxFoundation/strax>`_ framework.
Every simulation step is put into a dedicated plugin. This allows easy maintainance and
a flexible way to add new features. 

.. note::

   This project is under active development. Changes to the code and documentation
   are made on a regular basis.

.. toctree::
   :maxdepth: 1
   :caption: Setup and basics
   
   setup
   simulation_chain
   microphysics_simulation
   detector_physics
   pmt_and_daq

.. toctree::
   :maxdepth: 1
   :caption: Usage
   
   tutorials/Simulation_Refactor.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Microphysics Plugins
   
   plugins/micro_physics/ChunkInput
   plugins/micro_physics/FindCluster
   plugins/micro_physics/MergeCluster
   plugins/micro_physics/XENONnT_TPC
   plugins/micro_physics/XENONnT_BelowCathode
   plugins/micro_physics/VolumesMerger
   plugins/micro_physics/ElectricField
   plugins/micro_physics/NestYields
   plugins/micro_physics/MicrophysicsSummary

.. toctree::
   :maxdepth: 1
   :caption: Detector Physics Plugins
   
   plugins/detector_physics/S1PhotonHits
   plugins/detector_physics/S1PhotonPropagation
   plugins/detector_physics/ElectronDrift
   plugins/detector_physics/ElectronExtraction
   plugins/detector_physics/ElectronTiming
   plugins/detector_physics/SecondaryScintillation
   plugins/detector_physics/S2PhotonPropagation
   plugins/detector_physics/Delayed_Electrons

.. toctree::
   :maxdepth: 1
   :caption: PMT and DAQ Plugins
   
   plugins/pmt_and_daq/PMTAfterPulses
   plugins/pmt_and_daq/PhotonSummary
   plugins/pmt_and_daq/PMTResponseAndDAQ

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
