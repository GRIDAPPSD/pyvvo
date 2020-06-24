# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
# Add top-level of repository to the path for auto-documentation.
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'PyVVO'
copyright = '2020, Pacific Northwest National Laboratory and Brandon Thayer'
author = 'Brandon Thayer'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.mathjax',
              'sphinx.ext.extlinks', 'sphinx.ext.githubpages']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
#
# Added: latex, html, README.rst.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'html', 'latex',
                    'rst_latex', 'README.rst', '*.py']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Don't copy reST sources.
html_copy_source = False

########################################################################
# Prevent skipping __init___
# Source: https://stackoverflow.com/a/9772922
########################################################################

autoclass_content = 'both'

########################################################################
# Save some typing and automagically set some autodoc flags.
########################################################################
autodoc_default_options = {'members': True, 'undoc-members': True,
                           'show-inheritance': True}

########################################################################
# Mock up imports so that the documentation can be built in a simple
# virtual environment with only sphinx installed.
########################################################################

# Note: setup.py specifies mysqlclient, but really we import MySQLdb.
#
# Note: gridappsd is not listed in setup.py since it's included in the
# base Docker image for the application.
#
# Note: simplejson is not present here so that __init__.py can actually
# be run. All modules have been adjusted to fall back to the json
# package if simplejson cannot be found.
#
# Note: scikit-learn imports as sklearn
#
# Note: due to some constants in zip.py (and possibly elsewhere)
# depending on numpy, our documentation build environment should have
# numpy installed and NOT mocked.
autodoc_mock_imports = [
    'MySQLdb', 'mysqlclient', 'pandas', 'python-dateutil',
    'scipy', 'sklearn', 'stomp', 'gridappsd', 'dateutil']

########################################################################
# Configuration for extlinks.
########################################################################
# NOTES:
# The hicss link must be used with a '5' - a trailing slash breaks the
# link.
#
# The gld-wiki link must be used with a 's' - a trailing slash breaks
# the link.
#
# For links like gld-home or gld-github, simply pass a '/'
# character. Trailing slashes don't hurt here, and using extlinks allows
# a single definition of the URLs.
extlinks = {
    'hicss': ('http://hdl.handle.net/10125/6411%s', None),
    'gad-using': ('https://gridappsd.readthedocs.io/en/latest/using_gridappsd/index.html#%s', None),
    'gad-dev': ('https://gridappsd.readthedocs.io/en/latest/developer_resources/index.html#%s', None),
    'gld-home': ('https://www.gridlabd.org%s', None),
    'gld-wiki': ('http://gridlab-d.shoutwiki.com/wiki/Quick_link%s', None),
    'gld-github': ('https://github.com/gridlab-d/gridlab-d%s', None),
    'readme': ('https://github.com/GRIDAPPSD/pyvvo#%s', None)
}

########################################################################
# Configuration for linkcheck.
########################################################################
# The anchors in the README file come back as invalid. Unfortunate,
# but we have not other option but to skip.
linkcheck_ignore = ['https://github.com/GRIDAPPSD/pyvvo#', ]
