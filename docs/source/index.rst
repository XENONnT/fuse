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
   
   tutorials/Getting_Started.ipynb
   tutorials/Microphysics_Simulation.ipynb
   tutorials/Detectorphysics_Simulation.ipynb
   tutorials/csv_input.ipynb
   tutorials/Custom_Simulations.ipynb

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
