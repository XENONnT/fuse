=========================
Deterministic Random Seed
=========================

One of the main principles of fuse simulations are deterministic random seeds. The goal is to
make sure that the simulation is reproducible if the simulation is repeated under the same conditions.
This is important for debbuging and testing. Furthermore it allows us to store only some of the intermediate
simulation results and process other steps again in case we need them.

When should the simulation give reproducible results?
-----------------------------------------------------
The simulation should give the same results over and over if we have the same simulation conditions.
These conditions are:

- the same plugins
- the same plugin versions
- the same software versions
- the same config options
- the same run id

When any of these items change, we want to have different results.

How do we make the simulation reproducible?
--------------------------------------------
We can make the simulation reproducible by manually setting the random seed for all random number processes.
The random seed is a number that is used to initialize the random number generator. If we set the random seed
to the same number, the random number generator will always produce the same sequence of random numbers. The
random needs calculated from the list of conditions above. This is done using a hash function. The hash function
is deterministic, so it will always produce the same hash for the same input. The hash is then converted to a number
that is used as the random seed. The inputs to the hash function are the plugin lineage and the run id. The lineage
contains information about all used plugins, versions and config options. The run id is set by the user. Only config
options that are tracked by strax enter the lineage. The used hash function is implemented in the straxen package.


How do we make sure that the random seed is used everywhere?
-------------------------------------------------------------
There are three software packages that generate random numbers in fuse: numpy, scipy and nestpy. It is possible to set
the random seed for all three.

- **numpy**: To set the random seed for numpy we make use of numpy random Generator objects.
  Each plugin with numpy random random numbers has its own generator. the random seed of
  of the generator is set in the plugins setup function.

  .. code-block:: python

    self.rng = np.random.default_rng(seed = seed)

  When numpy random numbers are drawn inside a numba accelerated function, the generator needs to be passed as an argument.
  As random number generation in numba functions is not thread save, fuse can not use multi core processing.

- **scipy**: The random state of scipy random number generation can be set using the numpy random number generator
  object. The random state is set in the plugins setup function.

- **nestpy**: Setting the random seed for nestpy is possible since nestpy verison 2.0.1. The random number generator
  in nestpy is a C++ object and needs to be set once for the whole simulation. The random seed can then be set in the
  plugins setup function.

How to test if the simulation is reproducible?
-----------------------------------------------
If you run a simulation two times with the same config, run_id and plugins the output should be identical.
If you then change the run_id or a tracked config option, the simulation should give you different results.
fuse will print the random seeds of the plugins when `debug` is set to `True`. You can use this to check if the
random seeds are set correctly. If you build a new plugin, make sure to follow the deterministic random seed principle!
Addidtionaly we have a simple test for the deterministic random seed in the tests folder.
