# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


# -- Project information -----------------------------------------------------

project = 'csaps'
copyright = '2017-2020, Eugene Prilepin'
author = 'Eugene Prilepin'


def _get_version():
    about = {}
    ver_mod = ROOT_DIR / 'csaps' / '_version.py'
    exec(ver_mod.read_text(), about)
    return about['__version__']


# The full version, including alpha/beta/rc tags
release = _get_version()


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'matplotlib.sphinxext.plot_directive',
    'numpydoc',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

master_doc = 'index'

plot_apply_rcparams = True
plot_rcparams = {
    'axes.facecolor': "None",
    'figure.autolayout': "True",
    'savefig.bbox': "tight",
    'savefig.facecolor': "None",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    'show_powered_by': 'false',
    'github_user': 'espdev',
    'github_repo': 'csaps',
    'github_type': 'star',

    'description': 'Cubic spline approximation (smoothing)',

    'extra_nav_links': {
        'GitHub repository': 'https://github.com/espdev/csaps',
    },
}
