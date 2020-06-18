pyvvo
=====

This directory contains all of PyVVO's application code. This README
does not seek to be comprehensive, but rather to give a high-level
overview of the purpose of each file/module.

\__init__.py
-------------

Standard file for signifying a Python package. PyVVO's ``__init__.py``
file also sets up application-level logging, reading configuration
fields from ``log_config.json``.

app.py
------

This module contains PyVVO's main application loop. When running inside
the GridAPPS-D platform, ``app.py`` is invoked by ``run_pyvvo.py``.

cluster.py
----------

Short module with a few functions for performing K-means clustering on
"smart meter" data that PyVVO uses.

db.py
-----

Short module containing helper functions for MySQL database interaction.

env.json
--------

File created by ``pyvvo/utils/create_env_file.py``. Used in PyCharm
run configurations paired with the "EnvFile" PyCharm plugin.

equipment.py
------------

Contains classes for representing power system equipment
(e.g., voltage regulators). There are also helper functions for
initializing many equipment objects at once or looping through
collections of equipment objects.

ga.py
-----

Contains all code for PyVVO's genetic algorithm.

glm.py
------

Contains functions and classes for managing GridLAB-D models (which are
``.glm`` files). The ``GLMManager`` class is what most users will be
after.

gridappsd_platform.py
---------------------

Contains helper functions and classes for interacting with the
GridAPPS-D platform.

load_model.py
-------------

Module for pulling data from the platform and creating load models.

log_config.json
---------------

Configuration file for PyVVO's logging. Used by ``__init__.py``

platform_config.json
--------------------

Configuration file for interfacing PyVVO with the GridAPPS-D platform.
For more information, see "Using GridAPPS-D"/"Hosting Application" in
the `GridAPPS-D documentation
<https://gridappsd.readthedocs.io/en/latest/index.html>`__.

pyvvo_config.json
-----------------

Configuration file for tweaking all sorts of PyVVO parameters. The
README file at the top level of this repository describes each field
in detail.

README.rst
----------

This file.

run_pyvvo.py
------------

Simple script to run ``app.py``. This script is invoked by the platform
itself, as can be seen in ``platform_config.json``.

sparql.py
---------

Contains a class for making SPARQL queries, as well as a large
collection of useful queries. The GridAPPS-D platform uses a
triple-store database to store Common Information Model (CIM) data
related primarily to power system models.

timeseries.py
-------------

Contains functions for parsing time series data that comes out of the
GridAPPS-D platform.

utils.py
--------

Contains a variety of helper functions for PyVVO that didn't belong in
any other module.

zip.py
------

Contains the code for fitting load data to a ZIP load model. The
docstring provides significant details, and is best viewed in PyVVO's
built documentation (pyvvo/docs/html/index.html).