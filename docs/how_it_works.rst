How PyVVO Works
===============

This section will provide an overview of how PyVVO works by walking
through various flow charts. At times, specific Python modules or
classes will be referenced. In addition to examining the source code
itself, the API is documented in the :ref:`pyvvo-code` section.

PyVVO is *highly* configurable. To keep this documentation maintainable
and avoid repeating information, the configurable fields will not be
covered in detail here. Instead, check out :readme:`this section
<configuring-pyvvo>` in the repository's top-level README file.

Additionally, PyVVO emits logs a significant amount of information as it
runs. Many of PyVVO's log messages are described in
:readme:`this section <viewing-pyvvo-logs-as-simulation-proceeds>`
of the top-level README.

Flow Chart Conventions
----------------------

.. include:: rst_latex/flow_conventions.rst

High Level Summary
------------------

At the highest level, PyVVO contains two important components:
data-driven predictive load modeling and a genetic algorithm. The
load models are layered onto a GridLAB-D model, and resulting GridLAB-D
simulation outputs are used in the genetic algorithm's optimization
process.

Main Loop
---------

.. include:: rst_latex/main_loop.rst

.. _load-modeling:

Load Modeling
-------------

.. include:: rst_latex/load_modeling.rst

.. _genetic-algorithm:

Genetic Algorithm
-----------------
