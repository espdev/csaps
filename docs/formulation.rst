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

In other words, **csaps** is a Python modified port of MATLAB `CSAPS <https://www.mathworks.com/help/curvefit/csaps.html>`_ function
that is an implementation of Fortran routine SMOOTH from `PGS <http://pages.cs.wisc.edu/~deboor/pgs/>`_
(originally written by Carl de Boor).
