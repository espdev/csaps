.. _internals:

.. currentmodule:: csaps

Internals
=========

Spline Representation
---------------------

The computed splines are represented by classes :class:`SplinePPForm` for univariate/multivariate
and :class:`NdGridSplinePPForm` for nd-gridded data.

The spline coefficients are stored in the numpy array. It is 2D 1xM array for univariate data,
2D NxM array for multivariate data and 2D/ND array (tensor-product of univariate spline coefficients)
for nd-gridded data.

Let's look at a simple example.

The multivariate data::

    x = [1, 2, 3, 4]
    y = [(1, 2, 3, 4), (5, 6, 7, 8)]

Compute the spline:

.. code-block:: python

    >>> s = csaps(x, y).spline

And print the spline info and coefficients array:

.. code-block:: python

    >>> print(s)

    SplinePPForm
      breaks: [1. 2. 3. 4.]
      coeffs: shape (2, 12)
      pieces: 3
      order: 4
      ndim: 2

    >>> print(s.coeffs)

    [[0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 2. 3.]
     [0. 0. 0. 0. 0. 0. 1. 1. 1. 5. 6. 7.]]

Each row in the array contain coefficients for all spline pieces for corresponding data.
In our case we have 2D Y-data with shape (2, 4) and have the coefficients array with shape (2, 12)

    - 2 rows: 2 dimensions in the data
    - 12 columns: 3 pieces of the cubic spline (4-order)

The spline pieces for each dimension are composed sequentially in the one row.
Such representation allows us to compute tensor-product for nd-grid data and evaluate splines
without superfluous manipulations and reshapes of the coefficients array.
