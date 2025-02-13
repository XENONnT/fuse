# XENON fuse

[![PyPI version shields.io](https://img.shields.io/pypi/v/xenon-fuse.svg)](https://pypi.python.org/pypi/xenon-fuse/)
[![Coverage Status](https://coveralls.io/repos/github/XENONnT/fuse/badge.svg)](https://coveralls.io/github/XENONnT/fuse)
[![Test package](https://github.com/XENONnT/fuse/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/XENONnT/fuse/actions/workflows/pytest.yml)
[![Readthedocs Badge](https://readthedocs.org/projects/fuse/badge/?version=latest)](https://xenon-fuse.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/XENONnT/fuse/main.svg)](https://results.pre-commit.ci/latest/github/XENONnT/fuse/main)
[![DOI](https://zenodo.org/badge/622956443.svg)](https://zenodo.org/doi/10.5281/zenodo.11059395)

**F**ramework for **U**nified **S**imulation of **E**vents

fuse is the refactored version of the XENONnT simulation chain. The goal of this project is to unify [epix](https://github.com/XENONnT/epix) and [WFSim](https://github.com/XENONnT/WFSim) into a single program. fuse is based on the [strax framework](https://github.com/AxFoundation/strax), so that the simulation steps are encoded in plugins with defined inputs and outputs. This allows for a flexible and modular simulation chain.

## Installation

With all requirements fulfilled (e.g., on top of the [XENONnT montecarlo_environment](https://github.com/XENONnT/montecarlo_environment)):
```
python -m pip install xenon-fuse
```
or install from source:
```
git clone git@github.com:XENONnT/fuse
cd fuse
python -m pip install . --user
```

## Plugin Structure

The full simulation chain in split into multiple plugins. An overview of the simulation structure can be found below.

![fuse plugin structure](docs/source/figures/fuse_simulation_chain.png)
