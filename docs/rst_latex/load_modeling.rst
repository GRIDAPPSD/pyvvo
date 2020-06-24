..  Note that all \ref{} commands here correspond to references in
    ../latex/load_modeling.tex.

Overview
^^^^^^^^

PyVVO uses "smart" meter data to create time varying predictive load
models. These models get layered onto a :gld-home:`GridLAB-D </>` model
for use in the :ref:`genetic-algorithm`.

While there are many, many types of load models, PyVVO attempts to keep
things simple and uses the ZIP load model. The ZIP load model represents
a load (or a collection of loads) as part constant impedance (Z), part
constant current (I), and part constant power (P). PyVVO uses the ZIP
load model for several reasons:

*   ZIP load models have a voltage dependency, which is important for
    model-based voltage control applications.
*   ZIP load models are included in as part of every distribution system
    simulator under the sun.
*   ZIP load models are physics-based, as opposed to "black box" models.
*   ZIP load models only have a handful of parameters, making
    curve-fitting less time consuming.

..  Note the 5 must be passed in to the HICSS link since a trailing
    slash breaks things.

For more details, please see our :hicss:`HICSS paper <5>`.

High-Level Flow Chart and PyVVO Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The image below depicts a flow chart of the operation of PyVVO's
load modeling. The entire procedure spans several modules (presented
here in alphabetical order):

*   :py:mod:`pyvvo.cluster` for data clustering operations
*   :py:mod:`pyvvo.gridappsd_platform` for pulling data from the
    GridAPPS-D platform
*   :py:mod:`pyvvo.load_model` for pulling all the pieces together from
    the different modules
*   :py:mod:`pyvvo.timeseries` for parsing and resampling raw data from
    the platform
*   :py:mod:`pyvvo.zip` for curve-fitting data to create ZIP load
    models

.. image:: latex/load_modeling.svg

TODO: keep writing.