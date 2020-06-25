.. _todo:

Outstanding Work, Known Issues, Future Work
===========================================
In progress.

-   More stopping conditions for ``GAStopper``, *e.g.*, load mismatch.
-   Load allocation loop
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
