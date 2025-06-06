.. _index:

csaps
=====

CSAPS -- Cubic Spline Approximation (Smoothing)

Version:
|release|

Overview
--------

**csaps** is a package for univariate, multivariate and nd-gridded data approximation using cubic smoothing splines.

The package provides functionality for computing and evaluating splines and can be useful
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

or using modern packaging tools like Poetry:

.. code-block:: bash

    poetry add csaps


The module depends only on NumPy and SciPy.

Python 3.10 or above is supported.

.. toctree::
    :caption: User Guide
    :hidden:

    formulation
    tutorial
    internals
    benchmarks
    changelog

.. toctree::
    :caption: API
    :hidden:

    api
    genindex

.. toctree::
    :caption: Project Links
    :hidden:

    GitHub <https://github.com/espdev/csaps>
    PyPI <https://pypi.org/project/csaps>
