# -*- coding: utf-8 -*-

"""
Univariate/multivariate cubic smoothing spline implementation

"""

import functools
from typing import Optional, Union

import numpy as np

import scipy.sparse as sp
import scipy.sparse.linalg as la
from scipy.interpolate import PPoly

from ._base import ISmoothingSpline
from ._types import UnivariateDataType, MultivariateDataType
from ._reshape import to_2d


class SplinePPForm(PPoly):
    """The base class for univariate/multivariate spline in piecewise polynomial form
    """


class CubicSmoothingSpline(ISmoothingSpline[SplinePPForm, float, UnivariateDataType]):
    """Cubic smoothing spline

    The cubic spline implementation for univariate/multivariate data.

    Parameters
    ----------

    xdata : np.ndarray, sequence, vector-like
        X input 1-D data vector (data sites: ``x1 < x2 < ... < xN``)

    ydata : np.ndarray, vector-like, sequence[vector-like]
        Y input 1-D data vector or ND-array with shape[axis] equal of `xdata` size)

    weights : [*Optional*] np.ndarray, list
        Weights 1-D vector with size equal of ``xdata`` size

    smooth : [*Optional*] float
        Smoothing parameter in range [0, 1] where:
            - 0: The smoothing spline is the least-squares straight line fit
            - 1: The cubic spline interpolant with natural condition

    extrapolate : [*Optional*] bool or 'periodic'
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. Default is True.

    axis : [*Optional*] int
        Axis along which ``ydata`` is assumed to be varying.
        Meaning that for x[i] the corresponding values are np.take(ydata, i, axis=axis).
        By default is -1 (the last axis).
    """

    def __init__(self,
                 xdata: UnivariateDataType,
                 ydata: MultivariateDataType,
                 weights: Optional[UnivariateDataType] = None,
                 smooth: Optional[float] = None,
                 extrapolate: Optional[Union[bool, str]] = None,
                 axis: int = -1):

        x, y, w, shape, axis = self._prepare_data(xdata, ydata, weights, axis)
        coeffs, self._smooth = self._make_spline(x, y, w, smooth, shape)
        self._spline = SplinePPForm.construct_fast(coeffs, x, extrapolate=extrapolate, axis=axis)

    def __call__(self, xi: UnivariateDataType) -> np.ndarray:
        """Evaluate the spline for given data
        """
        return self._spline(xi)

    @property
    def smooth(self) -> float:
        """Returns the smoothing factor

        Returns
        -------
        smooth : float
            Smoothing factor in the range [0, 1]
        """
        return self._smooth

    @property
    def spline(self) -> SplinePPForm:
        """Returns the spline description in `SplinePPForm` instance

        Returns
        -------
        spline : SplinePPForm
            The spline description in :class:`SplinePPForm` instance
        """
        return self._spline

    @staticmethod
    def _prepare_data(xdata, ydata, weights, axis):
        xdata = np.asarray(xdata, dtype=np.float64)
        ydata = np.asarray(ydata, dtype=np.float64)

        if xdata.ndim > 1:
            raise ValueError("'xdata' must be a vector")
        if xdata.size < 2:
            raise ValueError("'xdata' must contain at least 2 data points.")

        axis = ydata.ndim + axis if axis < 0 else axis

        if ydata.shape[axis] != xdata.size:
            raise ValueError(
                f"'ydata' data must be a 1-D or N-D array with shape[{axis}] "
                f"that is equal to 'xdata' size ({xdata.size})")

        # Rolling axis for using its shape while constructing coeffs array
        shape = np.rollaxis(ydata, axis).shape

        # Reshape ydata N-D array to 2-D NxM array where N is the data
        # dimension and M is the number of data points.
        ydata = to_2d(ydata, axis)

        if weights is None:
            weights = np.ones_like(xdata)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.size != xdata.size:
                raise ValueError('Weights vector size must be equal of xdata size')

        return xdata, ydata, weights, shape, axis

    @staticmethod
    def _compute_smooth(a, b):
        """
        The calculation of the smoothing spline requires the solution of a
        linear system whose coefficient matrix has the form p*A + (1-p)*B, with
        the matrices A and B depending on the data sites x. The default value
        of p makes p*trace(A) equal (1 - p)*trace(B).
        """

        def trace(m: sp.dia_matrix):
            return m.diagonal().sum()

        return 1. / (1. + trace(a) / (6. * trace(b)))

    @staticmethod
    def _make_spline(x, y, w, smooth, shape):
        pcount = x.size
        dx = np.diff(x)

        if not all(dx > 0):  # pragma: no cover
            raise ValueError(
                "Items of 'xdata' vector must satisfy the condition: x1 < x2 < ... < xN")

        dy = np.diff(y, axis=1)
        dy_dx = dy / dx

        if pcount == 2:
            # The corner case for the data with 2 points (1 breaks interval)
            # In this case we have 2-ordered spline and linear interpolation in fact
            yi = y[:, 0][:, np.newaxis]

            # FIXME: fix 'coeffs' array shape
            c = np.hstack((dy_dx, yi))
            p = 1.

            return c, p

        # Create diagonal sparse matrices
        diags_r = np.vstack((dx[1:], 2 * (dx[1:] + dx[:-1]), dx[:-1]))
        r = sp.spdiags(diags_r, [-1, 0, 1], pcount - 2, pcount - 2)

        dx_recip = 1. / dx
        diags_qtw = np.vstack((dx_recip[:-1], -(dx_recip[1:] + dx_recip[:-1]), dx_recip[1:]))
        diags_sqrw_recip = 1. / np.sqrt(w)

        qtw = (sp.diags(diags_qtw, [0, 1, 2], (pcount - 2, pcount)) @
               sp.diags(diags_sqrw_recip, 0, (pcount, pcount)))
        qtw = qtw @ qtw.T

        if smooth is None:
            p = CubicSmoothingSpline._compute_smooth(r, qtw)
        else:
            p = smooth

        pp = (6. * (1. - p))

        # Solve linear system for the 2nd derivatives
        a = pp * qtw + p * r
        b = np.diff(dy_dx, axis=1).T

        u = la.spsolve(a, b)
        if u.ndim < 2:
            u = u[np.newaxis]
        if y.shape[0] == 1:
            u = u.T

        dx = dx[:, np.newaxis]

        vpad = functools.partial(np.pad, pad_width=[(1, 1), (0, 0)], mode='constant')

        d1 = np.diff(vpad(u), axis=0) / dx
        d2 = np.diff(vpad(d1), axis=0)

        diags_w_recip = 1. / w
        w = sp.diags(diags_w_recip, 0, (pcount, pcount))

        yi = y.T - (pp * w) @ d2
        pu = vpad(p * u)

        c1 = np.diff(pu, axis=0) / dx
        c2 = 3. * pu[:-1, :]
        c3 = np.diff(yi, axis=0) / dx - dx * (2. * pu[:-1, :] + pu[1:, :])
        c4 = yi[:-1, :]

        c = np.vstack((c1, c2, c3, c4)).reshape((4, pcount - 1) + shape[1:])

        return c, p
