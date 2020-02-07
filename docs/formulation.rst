.. _formulation:

Smoothing Spline Formulation
============================

Definition
----------

The package implements cubic smooting spline algorithm proposed by Carl de Boor in his book
"A Practical Guide to Splines" [#]_.

The smoothing spline :math:`f` minimizes

.. math::

    p\sum_{j=1}^{n}w_j|y_j - f(x_j)|^2 + (1 - p)\int\lambda(t)|D^2f(t)|^2dt

where the first term is *error measure* and the second term is *roughness measure*.
Error measure weights :math:`w_j` are equal to 1 by default.
:math:`D^2f` denotes the second derivative of the function :math:`f`.

The smoothing parameter :math:`p` should be in range :math:`[0, 1]` where bounds are:
    - 0: The smoothing spline is the least-squares straight line fit to the data
    - 1: The natural cubic spline interpolant

The smoothing parameter p can be computed automatically based on the given data sites :math:`x`.

.. note::

    The calculation of the smoothing spline requires the solution of a linear system whose coefficient matrix
    has the form :math:`pA + (1 - p)B`, with the matrices :math:`A` and :math:`B` depending on the
    data sites :math:`X`. The automatically computed smoothing parameter makes ``p*trace(A) equal (1 - p)*trace(B)``.

Implementation
--------------

**csaps** is implemented as a pure (without C-extensions) Python modified port of MATLAB `CSAPS <https://www.mathworks.com/help/curvefit/csaps.html>`_ function
that is an implementation of Fortran routine SMOOTH from `PGS <http://pages.cs.wisc.edu/~deboor/pgs/>`_
(originally written by Carl de Boor). The implementation based on linear algebra routines and uses NumPy and sparse
matrices from SciPy.

Differences from smoothing splines in SciPy
-------------------------------------------

`scipy.interpolate <https://docs.scipy.org/doc/scipy/reference/interpolate.html>`_ package contains the following classes:

    - `UnivariateSpline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html>`_
    - `RectBivariateSpline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.html>`_
    - `SmoothBivariateSpline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.SmoothBivariateSpline.html>`_

These splines can be computed as :math:`k`-ordered (0-5) spline and its smoothing parameter :math:`s` specifies
the number of knots by specifying a smoothing condition. Also it is only univariate and rect bivariate (2D grid) splines.
The algrorithm cannot be used for vectorized computing splines for multivariate and nd-grid cases.

Also the performance of these splines depends on the data size and smoothing parameter ``s`` because
the number of knots will be iterative increased until the smoothing condition is satisfied::

    sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s

Unlike these splines the performance of **csaps** algorithm only depends on the data size and the data dimension.

**csaps** spline is cubic only and it has natural boundary condition type. The computation algorithm
is vectorized to compute splines for multivariate/gridded data. The smoothing parameter :math:`p` determines
the weighted sum of terms and limited by the range :math:`[0, 1]`. This is more convenient in practice
to control smoothing.

It is an example plot of comparison ``csaps`` and ``scipy.UnivariateSpline`` (k=3) with defaults (auto smoothing):

.. plot::
    :include-source: False

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import UnivariateSpline
    from csaps import CubicSmoothingSpline

    np.random.seed(1234)

    x = np.linspace(-5., 5., 25)
    y = np.exp(-(x/2.5)**2) + (np.random.rand(25) - 0.2) * 0.3
    xi = np.linspace(x[0], x[-1], 150)

    scipy_spline = UnivariateSpline(x, y, k=3)
    csaps_spline = CubicSmoothingSpline(x, y)

    yi_scipy = scipy_spline(xi)
    yi_csaps = csaps_spline(xi)

    plt.plot(x, y, 'o')
    plt.plot(xi, yi_scipy, '-', label='scipy UnivariateSpline')
    plt.plot(xi, yi_csaps, '-', label='csaps')
    plt.legend()


.. rubric:: Footnotes

.. [#] C. de Boor, A Practical Guide to Splines, Springer-Verlag, 1978 (`link <https://www.springer.com/gp/book/9780387953663>`_)
