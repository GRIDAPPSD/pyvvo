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