# XENON fuse 

**F**ramework for **U**nified **S**imulation of **E**vents

fuse is the refactored version of the XENONnT simulation chain. The goal of this project is to unify epix and WFSim into a single program. fuse is based on the strax software so that the simulation steps are encoded in plugins with defined inputs and outputs. This allows for a flexible and modular simulation chain.

fuse is still in an alpha-stage, so expect bugs and changes in the future.


![fuse setup](docs/source/setup.rst)


## Plugin Structure

The full simulation chain in split into multiple plugins. An overview of the simulation structure can be found below.

![fuse plugin structure](docs/source/figures/fuse_simulation_chain.png)

