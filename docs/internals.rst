.. _internals:

.. currentmodule:: csaps

Internals
=========

Spline Representation
---------------------

The computed splines are represented by classes :class:`SplinePPForm` (inherited from :class:`scipy.interpolate.PPoly`) for univariate/multivariate
and :class:`NdGridSplinePPForm` (inherited from :class:`scipy.interpolate.NdPPoly`) for n-d gridded data. This representation can be named as "PP-form"
(piecewise-polynomial form).
