.. _todo:

Outstanding Work, Known Issues, Future Work
===========================================

PyVVO is very nearly complete for a beta/v1 implementation. Progress was
significantly stymied by bugs in the GridAPPS-D sensor service,
timeseries API, and other things. Once historic data is available, PyVVO
can be wrapped up.

In short, load modeling needs to be inserted into :py:mod:`pyvvo.app`.
This will also involve work in :py:mod:`pyvvo.load_model`. The
``load_model.py`` module is currently (unfortunately) in a broken
state. There are also broken tests in ``test_load_model.py``.

You really should spend time reading the :ref:`load-modeling` section
of the documentation. There are some details there, too.

One of the first things you'll want to do is update the testing files.
See :ref:`testing`.


Misc Notes
----------

Brandon ran out of time, but here are a few things. You can also look
through the TODOs in the code, and you'll find a million things I'd like
to improve.

-   Find a suitable hosting solution for this documentation, or make
    GitHub pages work.
-   More stopping conditions for ``GAStopper``, *e.g.*, load mismatch.
-   Add a load allocation loop to get most accurate model.
-   Periodically attempt to determine if out-of-commission equipment
    is back in service.
-   Optimization loop termination, rather than run forever.
-   Optimization loop run periodically rather than as frequently as
    possible.
-   Load modeling process: go through with a fine-toothed comb and
    optimize it.
-   PyVVO currently only supports split-phase secondary load modeling.
-   Perhaps the load modeling curve-fitting should be primed with
    ZIP parameters from the previous fit. I would need to refresh my
    understanding of sequential least squares programming to think about
    whether or not this could get the modeling stuck in local minima.
-   Split out PyVVO's load modeling into its own application or
    service. This could easily be a "distributed app."
-   Control more devices, e.g. PV inverter voltage set points. This
    would involve work in ``equipment.py``, ``glm.py``, and ``ga.py``
    at the minimum.
