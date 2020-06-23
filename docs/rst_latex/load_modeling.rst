..  Note that all \ref{} commands here correspond to references in
    ../latex/load_modeling.tex.

The following image depicts a flow chart of the operation of PyVVO's
load modeling. The entire procedure spans several modules (presented
here in alphabetical order):

*   :py:mod:`pyvvo.cluster` for data clustering operations
*   :py:mod:`pyvvo.gridappsd_platform` for pulling data from the
    GridAPPS-D platform
*   :py:mod:`pyvvo.load_model` for pulling all the pieces together from
    top to bottom
*   :py:mod:`pyvvo.timeseries` for parsing and resampling raw data from
    the platform
*   :py:mod:`pyvvo.zip` for curve-fitting data to