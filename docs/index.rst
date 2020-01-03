csaps
=====

CSAPS -- Cubic Spline Approximation (Smoothing)

**csaps** is a package to approxime univariate, multivariate and n-dimensional gridded data
via cubic smoothing splines.

The package provides functionality for computing and evaluating splines.
It does not contain any spline analysis functions. Therefore, the package can be useful
in practical engineering tasks for data approximation and smoothing.

Installing
----------

You can install and update csaps using pip:

.. code-block:: bash

    pip install -U csaps

Python 3.5 or above is supported.

The module depends only on NumPy and SciPy.
On Windows we highly recommend to use unofficial builds
`NumPy+MKL <https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy>`_ and
`SciPy <https://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy>`_ from Christoph Gohlke.

Documentation
-------------

.. toctree::
    :maxdepth: 2

    formulation
    manual
    api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
