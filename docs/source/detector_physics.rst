Detector Physics Simulation
===========================

The detectorphysics simulation is performed in multiple plugins. They are listed below.

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
   plugins/detector_physics/delayed_electrons/PhotoIonizationElectrons
   plugins/detector_physics/delayed_electrons/DelayedElectronsDrift
   plugins/detector_physics/delayed_electrons/DelayedElectronsExtraction
   plugins/detector_physics/delayed_electrons/DelayedElectronsTiming
   plugins/detector_physics/delayed_electrons/DelayedElectronsSecondaryScintillation


.. image:: figures/DetectorPhysicsStructure.pdf
    :width: 600

.. toctree::
   :maxdepth: 1
   :caption: Alternative Plugins

   plugins/detector_physics/ChunkCsvInput

Alternatively the microphysics simulation can be skipped and only the detectorphysic simulation can be performed.
For this, the `ChunkCsvInput` plugin needs to be registered.
