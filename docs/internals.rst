.. _internals:

.. currentmodule:: csaps

Internals
=========

Spline Representation
---------------------

The computed splines are represented by classes :class:`SplinePPForm` (inherited from :class:`scipy.interpolate.PPoly`) for univariate/multivariate
and :class:`NdGridSplinePPForm` (inherited from :class:`scipy.interpolate.NdPPoly`) for n-d gridded data. This representation can be named as "PP-form"
(piecewise-polynomial form).

Here is an example of ``SplinePPForm`` object:

.. code-block:: python

    >>> from csaps import CubicSmoothingSpline

    >>> x = [0, 1, 2, 3]
    >>> y = [(1, 3, 4, 4), (5, 6, 7, 8)]

    >>> s = CubicSmoothingSpline(x, y)
    >>> print(s.spline)

    SplinePPForm
      breaks: [0. 1. 2. 3.]
      coeffs shape: (4, 3, 2)
      data shape: (2, 4)
      axis: 1
      pieces: 3
      order: 4
      ndim: 2

The coefficients array in this case has the shape ``(4, 3, 2)`` where:

    - 4 -- spline order (4 for cubic)
    - 3 -- the number of spline pieces (3 pieces for 4 X-points)
    - 2 -- the number of 1-d Y-data vectors
