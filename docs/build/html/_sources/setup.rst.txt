Setting up fuse
===============

To install fuse you need to make sure to have the latest version of nestpy installed. 
You can install nestpy from the source code by running the following commands:

.. code-block:: console

    $ git clone https://github.com/NESTCollaboration/nestpy
    $ cd nestpy
    $ git submodule update --init --recursive
    $ pip install .

Then its time to install fuse. All other dependencies will be installed automatically.

.. code-block:: console

    $ git clone https://github.com/XENONnT/fuse.git
    $ cd fuse
    $ pip install .