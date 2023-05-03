# XENON fuse 

**F**ramework for **U**nified **S**imulation of **E**vents

fuse is the refactored version of the XENONnT simulation chain. The goal of this project is to unify epix and WFSim into a single program. fuse is based on the strax software so that the simulation steps are encoded in plugins with defined inputs and outputs. This allows for a flexible and modular simulation chain.

fuse is still in an alpha-stage, so expect bugs and changes in the future.

## Installation

At the moment the intallation procedure is not very advanced. I would recommend to work on dali in e.g. the base environment and follow the steps below.

1. Clone the fuse repository.
2. Clone the private_nt_aux_files repository to the same directory as you cloned fuse.
3. Install fuse using `pip install -e .` in the fuse directory.


## Plugin Structure

The full simulation chain in split into multiple plugins. An overview of the simulation structure can be found below.

![Simulation_Refactor_Plugins](https://user-images.githubusercontent.com/27280678/235156990-7fa63aae-21c4-45b5-9b71-a42a4173f0da.jpg)

