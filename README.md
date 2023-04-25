# XeSim
Refactor XENONnT epix and WFsim code using the strax framework. 

## Installation

At the moment the intallation procedure is not very advanced. I would recommend to work on dali in e.g. the base environment and follow the steps below.

1. Clone the XeSim repository.
2. Clone the private_nt_aux_files repository to the same directory as you cloned XeSim.
3. Install XeSim using `pip install -e .` in the XeSim directory.


## Plugin Structure

The full simulation chain in split into multiple plugins. An overview of the simulation structure can be found below.

![Simulation_Refactor_Plugins](https://user-images.githubusercontent.com/27280678/233602026-a91c4ae0-a3ff-4fc9-ac08-2d725bbe8833.jpg)


