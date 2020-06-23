Building the PyVVO Documentation
================================

This section discusses what is required to build PyVVO's documentation.
TL;DR (though please read): ``python build_docs.py --checkout``

.. _venv:

Virtual Environment
-------------------

To keep things simple, Brandon builds the documentation via a Python
interpreter within a local virtual environment, rather than using
a Docker-based interpreter. The `Python docs
<https://docs.python.org/3/tutorial/venv.html>`__ provide a nice
tutorial for building and using virtual environments. TL;DR:

.. code:: bash

    cd ~/git/pyvvo
    python3 -m venv venv
    source venv/bin/activate

After activating the virtual environment, install ``sphinx`` and
``numpy``. The original goal was to *only* require installing ``sphinx``
in the virtual environment, but ``zip.py`` has quite a few constants
that depend on ``numpy``, so `mocking
<https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_mock_imports>`__
the ``numpy`` import simply isn't practical. Install like so:

.. code:: bash

    python -m pip install sphinx numpy

Initial Setup
-------------

The files ``conf.py``, ``index.rst``, ``make.bat``, and ``Makefile``
were originally created using the `sphinx-quickstart
<https://www.sphinx-doc.org/en/master/usage/quickstart.html>`__ tool.
All these files have since been modified.

Building the Documentation
--------------------------

Building the documentation is wicked simple. After activating your
virtual environment (see :ref:`venv`), and assuming your shell's
current working directory is ``pyvvo/docs``, simply:

.. code:: bash

    python build_docs.py --checkout

For details on the available flags/arguments, run:

.. code:: bash

    python build_docs.py --help

Note that all tools (``latex``, ``dvisvgm``, ``sphinx-build``) have
their outputs "quieted." Thus, if you get any output besides the
obvious output from the Python script itself, that's cause for concern.
For reference, with no warnings, clean output from ``build_docs.py``
should look like:

    ********************************************************************************
    Running tex2svg for main_loop
    Done.
    ********************************************************************************
    ********************************************************************************
    Building the documentation.
    Done.
    ********************************************************************************

Viewing the Built Documentation
-------------------------------

Simply open ``~/git/pyvvo/docs/html/index.html`` in your web browser.