..  Note that all \ref{} commands here correspond to references in
    ../latex/main_loop.tex.

The following image depicts a flow chart of the operation of ``app.py``
(:py:mod:`pyvvo.app`). Boxes prefaced with **INCOMPLETE** indicate that
more work is needed to finalize the code related to the process
described in the box.

.. image:: latex/main_loop.svg

As noted in \ref{flow:start}, when PyVVO is running inside the
GridAPPS-D platform, it's started by ``run_pyvvo.py``
(:py:mod:`pyvvo.run_pyvvo`).

.. _init-phase:

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

`Subscribing to simulation output <sim-output_>`_ is like drinking from
a fire hose, so PyVVO has the ``SimOutRouter`` class
(:py:class:`pyvvo.gridappsd_platform.SimOutRouter`) which filters
simulation output and calls methods of the equipment classes that keep
their states up to date \ref{flow:subscribe}. All state
updates/subscriptions occur in their own threads, so object states are
immediately updated whenever new measurements come in.

PyVVO uses `GridLAB-D <gld-home_>`_ (`wiki <gld-wiki_>`_,
`GitHub <gld-github_>`_) as its power flow solver/simulator, and the
GridAPPS-D platform is capable of creating a GridLAB-D model from the
CIM triplestore for its own simulations. PyVVO leverages this fact and
`requests a model <gld-base_>`_ of the power system in GridLAB-D
(``.glm``) format \ref{flow:pull-gld}, representing the nominal state
of the system.

Next, PyVVO initializes a ``GLMManager``
(:py:class:`pyvvo.glm.GLMManager`) \ref{flow:init-glm-mgr} using the
``.glm`` file pulled in \ref{flow:pull-gld}. The ``GLMManager``
creates an in-memory representation of the model using Python data
types, and is capable of modifying the model and writing out a new
``.glm`` file. The module :py:mod:`pyvvo.glm` could come in handy for
other GridAPPS-D applications, or any application that needs to read,
modify, and write GridLAB-D models. The code isn't perfect and has some
shortcomings, but also has a *lot* of features and functionality.

Next, PyVVO begins the process of load modeling by pulling historic
meter data from the GridAPPS-D platform's timeseries database
\ref{flow:pull-load-data}. Specifically, historic data should come from
the platform's `sensor service <sensor-data_>`_. As discussed in
:ref:`todo` and noted in the flow chart, this portion of PyVVO is
currently incomplete due to platform issues.

Weather data is incorporated in PyVVO's load modeling process. This
data is obtained by `querying the platform <weather-data_>`_
\ref{flow:pull-weather-data}. Once obtained, the weather data must
be parsed and resampled so that it matches up 1:1 with the meter data
in \ref{flow:pull-load-data}. See
:py:func:`pyvvo.timeseries.parse_weather` and
:py:func:`pyvvo.timeseries.resample_timeseries`.

.. _opt-phase:

Optimization Phase
^^^^^^^^^^^^^^^^^^
First of all, note there is some overlap in the "Initialization" and
"Optimization" phases due to loop priming.

After all procedures described in :ref:`init-phase` have been completed,
PyVVO enters its optimization loop. The first step in this process is
to update PyVVO's internal ``.glm`` model of the power system with the
current states of all equipment \ref{flow:update-glm-mgr}. Future
work might use predicted future states rather than current states.

Next, PyVVO initializes all the required objects for running the genetic
algorithm \ref{flow:init-ga}. The genetic algorithm components are fully
encapsulated in :py:mod:`pyvvo.ga`. As noted in the flow chart, a
``GAStopper`` object is used for stopping the genetic algorithm
prematurely based on system events. At present, this only includes
switches changing state (though `future work <todo>` should include
other conditions). If the genetic algorithm were not halted due to a
topology change, it would continue optimizing, but its internal model
would no longer represent reality, making the result wrong/suboptimal
(and perhaps dangerous to the health of the system).

The internal workings of the genetic algorithm itself are rather
complex, so discussion here will be limited to what's already mentioned
in \ref{flow:run-ga}. More details can be found in
:ref:`genetic-algorithm`.

After the genetic algorithm has run to completion, PyVVO has an idea of
how controllable devices should be operated (*e.g.*, regulators and
capacitors). These proposed device settings are sent into the platform
as commands (*e.g.* put regulator 1, phase A at tap 3)
\ref{flow:send-commands}. PyVVO only sends in commands for proposed
states that are different from the current state. In this way, if the
proposed states are identical to the present states (indicating the
system is already in an "optimal" configuration), no commands are sent
into the system.

Note that \ref{flow:check-done} is flagged with **INCOMPLETE**. This is
because at present, PyVVO simply runs the optimization loop indefinitely
and only stops when its Docker container is killed. PyVVO doesn't
presently require any particular clean-up procedures, hence why
\ref{flow:stop} is so simple.
