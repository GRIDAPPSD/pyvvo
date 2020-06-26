Brandon ran out of time to write about the genetic algorithm in detail,
but fortunately there's this nifty flow chart:

.. image:: latex/genetic_algorithm.svg

Note that many genetic algorithm parameters are controlled in
``pyvvo_config.json``. Specifically, have a look at the ``ga``,
``limits``, and ``costs`` keys. Discussion of these parameters is
covered in the :readme:`README <configuring-pyvvo>`.

It's pretty easy to follow along in the code with the flow chart in
hand. If you're using PyCharm, ``Ctrl + B`` is going to be your best
friend for jumping into methods as you follow the rabbit hole. As
seen in \ref{flow:init-ga} and \ref{flow:start-ga}, the algorithm is
kicked off in ``app.py``. From there, you can sift through ``ga.py``
to find everything you need.

Note that right now, stopping the algorithm is ill-defined
\ref{flow:stop}. It just runs for a fixed number of generations.
Ideally, some sort of convergence criteria would be developed.