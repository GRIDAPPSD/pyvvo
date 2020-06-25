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

Flow Chart and PyVVO Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

The procedures described in the flow chart above can be kicked off by
instantiating a :py:class:`pyvvo.load_model.LoadModelManager` object
and then calling the ``fit_for_all`` method. Note that the meta data
described in \ref{flow:pull-meta-data} are required inputs for
initializing a ``LoadModelManager``. Also note that the ``load_model``
module and associated code are in need of some more work, testing,
*documentation*, and attention.

Generally speaking, the flow chart above speaks for itself. However,
some items that are worth elaborating on:

-   As per usual, many of the parameters (*e.g.* :math:`M_{cs}`) are
    configurable. If the parameters aren't already in ``pyvvo.json``,
    they can easily be added and
    :readme:`documented <configuring-pyvvo>`.
-   At the time of writing, the data standardization in
    \ref{flow:standard-scale} scale is unnecessarily done inside the
    clustering and fitting loop. With some moderate refactoring, this
    can be fixed.
-   K-means clustering in \ref{flow:cluster} is performed via
    Scikit-learn's `implementation
    <https://scikit-learn.org/stable/modules/clustering.html#k-means>`__.
-   Note that voltage (:math:`V`) is not used in clustering, because
    clustering by voltage will lead to a poorly performing ZIP model.
    The ZIP load model takes voltage magnitude as input and outputs
    :math:`P` and :math:`Q`. Therefore, it behooves us to have diverse
    voltage values in the data used for fitting, which would be undercut
    by including voltage as a clustering feature.
-   In \ref{flow:select-cluster}, weather data is exclusively used as
    cluster selection features. This is a heuristic assumption based on
    the load/weather dependence that all power systems engineers are
    well aware of. This has not necessarily been proven to be the best
    approach in all scenarios, and in fact we've seen that at times a
    better fit can be obtained by including :math:`P` and :math:`Q` in
    the cluster selection.
-   Similarly, we do not have a guarantee that our "best" cluster
    necessarily leads to the "best" fit. However, it would be very
    computationally expensive to perform a ZIP fit on every cluster.
    On the other hand, maybe that approach would be worthwhile. There
    is a wealth of experimentation that can still be done.
-   The least squares optimization routine mentioned in
    \ref{flow:ls-fit} is moderately complex. It's worth exploring this
    code in detail. Simply follow the rabbit hole starting with
    :py:func:`pyvvo.zip.zip_fit`.
-   The final ZIP model that comes out of \ref{flow:ls-fit} uses
    GridLAB-D conventions. A thorough examination of the GridLAB-D
    source code was undertaken to ensure exact concurrence. There are
    tests to prove it in ``tests/test_zip.py``.
-   The reason for using a normalized MSE in \ref{flow:norm-mse} is that
    different clustering loop runs will have a different number of
    data points present in the "best" cluster, resulting in very
    different raw MSE values.
-   Note that equal weight is given to :math:`\text{MSE}_P` and
    :math:`\text{MSE}_Q` in the :math:`\text{MSE}_\text{norm}`
    computation in \ref{flow:norm-mse}. It may be valuable to experiment
    with different weighting schemes.

Outstanding Work and Possible Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The GridAPPS-D platform has struggled to put together a working sensor
service, and the timeseries database and its API have been plagued by
bugs. Additionally, memory leaks and other issues have prevented the
platform from running for a prolonged time to generate historic data
for the load modeling. As such, PyVVO's load modeling procedures have
not been fully integrated into ``app.py`` and ``load_model.py`` has
several outstanding issues.

Fortunately, the underlying clustering and fitting code is complete (at
least in a working and tested draft form), and most of the remaining
work involves finalizing the touch points between PyVVO and the platform
with respect to load modeling.

It would be prudent to walk through the entire clustering and fitting
process with an eye toward performance optimization. There are **many**
loads in the so called "9500 node" model (~1300) and getting the
required data for each load requires 4 different measurement objects
(as mentioned in \ref{flow:pull-load-data} in the flow chart).

I (Brandon) suspect that the load modeling procedure is going to run
into bottlenecks with respect to both I/O from the platform as well as
computation.

.. _io-issues:

**With respect to I/O**: queries to the timeseries database
have to date been very slow, and message size and/or memory issues
means that it may be necessary to perform a single query for each load
(or maybe even for each measurement!) which comes with a **lot** of
overhead. Additionally, due to the primitive filtering available through
the GridAPPS-D API, there are two options for time filtering:

1.  Pull all historic data at once (*e.g.*, all data for a two week
    window) and filter it afterwards. Ultimately, PyVVO is only going
    to use something like < 1/10 of the pulled data, so this is clearly
    inefficient.
2.  Perform *lots* of little queries for the the various time windows.
    *E.g.*, perform ten queries to pull data from 9:00am-11:00am for
    two weeks worth of weekday data. This clearly comes with a lot of
    overhead.

While it may go against the API-only "principles" of GridAPPS-D, the
best solution would be to query the timeseries database directly and
create moderately complex custom time filters. This could be done
through the API if a "custom query" route was created, similarly to
the :gad-using:`SPARQL API <query>`.

It's also worth noting that if the historic data is ever generated at
the correct averaging interval by the sensor service (*e.g.*,
15 minutes), all the load data for a particular time window *might* fit
into memory with a single query. Again, you're likely going to run into
maximum message size issues with the platform, although the database
itself would be totally happy to hand you all that data at once.

Finally, some thoughts on parallelization: depending on how the platform
API and database infrastructure are implemented, querying the database
in parallel on the application side may not result in truly parallel
queries on the platform side. *E.g.*, if the database queries are
multi-threaded instead of multi-processed, you won't actually get true
concurrency, just the "fake" concurrency that threading provides.

.. _computation-issues:

**With respect to computation**: The bottom line is there are a lot of
loads, and for each load multiple clustering operations and sequential
least squares optimization operations are run. That's a lot of
computation. The good news is that this is **completely**
parallelizable. GridAPPS-D as a project is emphasizing distributed
applications, so splitting out the load modeling into its own app could
be a very valuable use case. This would also be useful for other
applications that rely on load models, such as WSU's VVO application.

There are also a lot of tweaks that can be made to potentially speed
up the load modeling process. For example, increasing the minimum
cluster size :math:`M_{cs}` in \ref{flow:compute-max-clusters} will
decrease the number of clustering loops that are performed, at the cost
of less exploration. Additionally, the least-squares optimization could
potentially be sped up by using the previous fit parameters as a
starting point for the next optimization run. There are likely lots of
other little levers such as these that could help alleviate the
computation bottleneck.

If you're feeling lazy and have the computational resources, just throw
more cores at the problem. However, depending on how you get the data,
you might run into I/O bottlenecks (as discussed in the
:ref:`IO issues <io-issues>` section.

If you are feeling really blasphemous and are okay flying in the face of
all the load modeling work that's been done for this application, you
could do something as simple (and likely very, very suboptimal) as use
state estimator :math:`P` and :math:`Q` output for each load and model
them all as constant power (or constant current or constant impedance
or with arbitrary ZIP parameters). Please don't do this.

Yet another intriguing option would be to aggregate loads up to the
distribution transformer level. In short, one would need to model the
voltage drop and losses across the "triplex" lines, and then aggregate
at the historic data at transformer level. This aggregated data could
then go through the same ZIP fit procedure described here. However, you
lose some important information: what's the voltage at each meter? It's
feasible that downstream of one secondary transformer there's one meter
within the allowed voltage band and one outside the allowed voltage
band. How often does this happen? Hard to say.