.. _testing:

Testing
=======

PyVVO has a great many tests. This is necessary because, well, you
should always test your software. Also, it's a complex application,
and with any complex application you can't trust that it's working
without tests. This is also useful if a code modification is made -
the tests help ensure that something didn't get inadvertently broken
(though there's always room for hidden bugs that the tests don't
catch!). These tests have also caught countless platform bugs, and have
thus proven their worth in more than one dimension.

There are mentions of the tests in a couple places in the README:
:readme:`here <run-the-tests>` and :readme:`here <run-the-tests-1>` -
please check those out. The information there will not be duplicated
here.

The PyVVO tests contain all sorts of tests: unittests, integration
tests, and regression tests. The only bad thing is that these are not
all marked as such, and are all mixed together. Sorry about that.

Right now (2020-06-25), there are going to be a fair number of tests
that fail. Some of this is related to platform issues, and part of this
is related to some unfinished code (especially in
:py:mod:`pyvvo.load_model` - see :ref:`load-modeling` for some more
information).

So as to avoid calling the platform for all tests, there are a lot of
testing files that contain expected output from the platform. These
need updated from time to time. There are two scripts for updating the
test data files: `tests/data_files.py` and `tests/models.py`. I'm not
going to claim these scripts are super clean and super awesome in every
way, but they get the job done.

`tests/data_files.py` has a bunch of different functions for generating
data files. Many of these involve running simulations or performing
lots of queries. It's not recommended to run all the functions at once -
rather, run a function at a time, check the diff for the data file,
commit it, then move on to the next file. This is an unfortunately
manual, time-consuming, and boring process. Improvements are always
welcome.

`tests/models.py` is simpler, and essentially just queries the platform
to pull GridLAB-D models. Don't hesitate to just run it. It's always
a good idea to check the diffs before committing changes to testing
files.

Unfortunately, some regression tests will fail from time to time since
the platform's CIM triplestore database isn't deterministic - data
doesn't always come back in the same order! That's an outstanding action
item: update the testing code to sort DataFrames before comparing them.