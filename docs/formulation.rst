.. _formulation:

Smoothing spline formulation
============================

The package implements cubic smooting spline algorithm proposed by Carl de Boor in his book
"A Practical Guide to Splines" (`Springer <https://www.springer.com/gp/book/9780387953663>`_).

The smoothing spline :math:`f` minimizes

.. math::

    p\sum_{j=1}^{n}w_j|y_j - f(x_j)|^2 + (1 - p)\int\lambda(t)|D^2f(t)|^2dt

where the first term is *error measure* and the second term is *roughness measure*.
Error measure weights :math:`w_j` are equal to 1 by default.
:math:`D^2f` denotes the second derivative of the function :math:`f`.

The smoothing parameter :math:`p` should be in range :math:`[0, 1]` where bounds are:
    - 0: The smoothing spline is the least-squares straight line fit to the data
    - 1: The natural cubic spline interpolant

By deafult, the smoothing parameter p is computed automatically based on the given data sites :math:`x`.

**csaps** is implemented as a Python modified port of MATLAB `CSAPS <https://www.mathworks.com/help/curvefit/csaps.html>`_ function
that is an implementation of Fortran routine SMOOTH from `PGS <http://pages.cs.wisc.edu/~deboor/pgs/>`_
(originally written by Carl de Boor).

Differences from SciPy UnivariateSpline
---------------------------------------

`scipy.interpolate <https://docs.scipy.org/doc/scipy/reference/interpolate.html>`_ package contains
`UnivariateSpline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html>`_ class.

This spline can be computed as :math:`k`-ordered (0-5) spline and its smoothing parameter :math:`s` specifies
the number of knots by specifying a smoothing condition. Also it is only univariate spline.
The algrorithm cannot be used for vectorized computing splines for multivariate and gridded cases.

**csaps** spline is cubic only and it has natural boundary condition type. The computation algorithm
is vectorized to compute splines for multivariate/gridded data. The smoothing parameter :math:`p` determines
the weighted sum of terms and limited by the range :math:`[0, 1]`. This is more convenient in practice
to control smoothing.
