..  Note that all \ref{} commands here correspond to references in
    ../latex/main_loop.tex.

The following image depicts a flow chart of the operation of ``app.py``.
Boxes prefaced with **INCOMPLETE** indicate that more work is needed
to finalize the code related to the process described in the box.

.. image:: latex/main_loop.svg

As noted in \ref{flow:start}, when PyVVO is running inside the
GridAPPS-D platform, it's started by ``run_pyvvo.py.``

Initialization Phase
^^^^^^^^^^^^^^^^^^^^

When PyVVO is started, it only receives two inputs from
the platform: the simulation ID and the `simulation request
<https://gridappsd.readthedocs.io/en/latest/using_gridappsd/index.html#simulation-api>`__
\ref{flow:sim-request}. The simulation request contains many useful
details including, but not limited to, the feeder's MRID, the time span
of the simulation, *etc.*

PyVVO uses the information from \ref{flow:sim-request} to initialize a
variety of classes whose role is to interface with the GridAPPS-D
platform \ref{flow:init-interfaces}. These classes can be found in
:py:mod:`pyvvo.sparql` and :py:mod:`pyvvo.gridappsd_platform`.

Since PyVVO is a volt-var optimization application, it primarily cares
about regulators and capacitors (future work should include control of
other devices). However, since PyVVO is model-based, it also needs to
know about other active devices in the system such as photovoltaic
inverters, distributed generators, and switches. PyVVO uses the
``SPARQLManager`` (initialized in \ref{flow:init-interfaces}) to query
the GridAPPS-D platform's Common Information Model (`CIM`_)
triplestore database to obtain nominal device information
\ref{flow:init-interfaces}. Additionally, information about the
measurement objects (*e.g.*, their MRIDs, measurement types, *etc*.)
associated with the each device is pulled from the CIM triplestore.

With device and measurement information in hand, PyVVO can initialize
objects that represent all the equipment in the system that PyVVO cares
about \ref{flow:eq-mgrs}. PyVVO has various classes related to the
management of devices (a.k.a. "equipment") in :py:mod:`pyvvo.equipment`.
These classes generally contain a small subset of what's contained in
the `CIM`_.

`Subscribing to simulation output <sim-output>_`_ is like drinking from
a fire hose, so PyVVO has the ``SimOutRouter`` class
(:py:class:`pyvvo.gridappsd_platform.SimOutRouter`) which filters
simulation output and calls methods of the equipment classes that keep
their states up to date \ref{flow:subscribe}.

.. _CIM: https://gridappsd.readthedocs.io/en/latest/developer_resources/index.html#cim-documentation
.. _sim-output: https://gridappsd.readthedocs.io/en/latest/using_gridappsd/index.html#subscribe-to-simulation-output