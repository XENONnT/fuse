# XENON fuse 

**F**ramework for **U**nified **S**imulation of **E**vents

fuse is the refactored version of the XENONnT simulation chain. The goal of this project is to unify epix and WFSim into a single program. fuse is based on the strax software so that the simulation steps are encoded in plugins with defined inputs and outputs. This allows for a flexible and modular simulation chain.

## Installation

1. Clone the fuse repository.
2. Install fuse using `pip install . --user` in the fuse directory.

## Plugin Structure

The full simulation chain in split into multiple plugins. An overview of the simulation structure can be found below.

![fuse plugin structure](docs/source/figures/fuse_simulation_chain.png)

