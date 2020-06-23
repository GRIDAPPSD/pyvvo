docs
====

This directory contains both source and built files for PyVVO's
documentation. To read the built documentation, simply open
``html/index.html`` in your web browser. This built documentation
contains directions for building the documentation when it needs
to be updated.

Files and Directories
---------------------

The ``*.rst`` files in this directory should be included by
``index.rst``. As such, this README will not discuss each individual
``*.rst`` file.

_static
^^^^^^^

This directory houses static files required for building PyVVO's
documentation. At the moment, there are none.

_templates
^^^^^^^^^^

This directory houses templates required for building PyVVO's
documentation. At the moment, there are none.

html
^^^^

This directory contains the built HTML documentation for PyVVO. Open
``index.html`` in your web browser to view the docs.

latex
^^^^^

This directory contains ``.tex`` source files for diagrams included in
PyVVO's documentation. These ``.tex`` files are compiled into ``.dvi``
files via ``latex`` and then converted to ``.svg`` via the ``dvisvgm``
command line tool. See the README in the ``latex`` directory for more
details.

rst_latex
^^^^^^^^^

This directory contains ``.rst`` files that should correspond directly
(and have an identical name, except for the extension) to ``.tex`` files
in the ``latex`` directory.

conf.py
^^^^^^^

Configuration file used by Sphinx for building PyVVO's documentation.

index.rst
^^^^^^^^^

Main reStructuredText file for PyVVO's documentation. All other
``*.rst`` files are included by it.

Makefile
^^^^^^^^

Makefile for building PyVVO's documentation, making building as simple
as ``make html``

README.rst
^^^^^^^^^^

This file.
