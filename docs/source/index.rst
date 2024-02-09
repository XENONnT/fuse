Welcome to the documentation of XENON fuse!
===========================================

**fuse** is the next generation XENONnT simulation software. The goal of this project is
to unify `epix <https://github.com/XENONnT/epix>`_ and `WFSim <https://github.com/XENONnT/WFSim>`_ into a
single program. fuse is based on the `strax <https://github.com/XENONnT/straxen>`_, so that the simulation
steps are encoded in plugins with defined inputs and outputs. This allows for a flexible and modular
simulation chain.

.. toctree::
   :maxdepth: 1
   :caption: Getting started
   
   installation
   tutorials/0_Getting_Started.ipynb
   tutorials/1_Microphysics_Simulation.ipynb
   tutorials/2_Detectorphysics_Simulation.ipynb
   tutorials/3_csv_input.ipynb
   tutorials/4_Custom_Simulations.ipynb
   

.. toctree::
   :maxdepth: 1
   :caption: Simulation Plugins
   
   simulation_chain
   microphysics_simulation
   detector_physics
   pmt_and_daq

.. toctree::
   :maxdepth: 1
   :caption: Technical Features
   
   tech_features/DeterministicSeed
   tech_features/DynamicChunking

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`