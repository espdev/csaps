csaps
=====

CSAPS -- Cubic Spline Approximation (Smoothing)

**csaps** is a package to approxime univariate, multivariate and n-dimensional gridded data
via cubic smoothing splines.

The package provides functionality for computing and evaluating splines.
It does not contain any spline analysis functions. Therefore, the package can be useful
in practical engineering tasks for data approximation and smoothing.

Here is an example of the univariate data smoothing:

.. code-block:: python
    :linenos:
    :emphasize-lines: 2,8

    import numpy as np
    from csaps import csaps

    x = np.linspace(0., 2*np.pi, 25)
    y = np.sin(x) + np.random.randn(25) * 0.3
    xi = np.linspace(x[0], x[-1], 151)

    yi = csaps(x, y, xi, smooth=0.8)


Installing
----------

You can install and update csaps using pip:

.. code-block:: bash

    pip install -U csaps

The module depends only on NumPy and SciPy.
Python 3.5 or above is supported.


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
