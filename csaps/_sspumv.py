# -*- coding: utf-8 -*-

"""
Univariate/multivariate cubic smoothing spline implementation

"""

import typing as ty
import functools

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la

from ._base import SplinePPFormBase, ISmoothingSpline
from ._types import UnivariateDataType, MultivariateDataType
from ._reshape import from_2d, to_2d


class SplinePPForm(SplinePPFormBase[np.ndarray, int]):
    """Univariate/multivariate spline representation in PP-form

    Parameters
    ----------
    breaks : np.ndarray
        Breaks values 1-D array

    coeffs : np.ndarray
        Spline coefficients 2-D array

    """

    def __init__(self, breaks: np.ndarray, coeffs: np.ndarray) -> None:
        self._breaks = breaks
        self._coeffs = coeffs
        self._pieces = breaks.size - 1
        self._order = coeffs.shape[1] // self._pieces
        self._ndim = coeffs.shape[0]

    @property
    def breaks(self) -> np.ndarray:
        """Returns the breaks array"""
        return self._breaks

    @property
    def coeffs(self) -> np.ndarray:
        """Returns the spline coefficients 2-D array"""
        return self._coeffs

    @property
    def order(self) -> int:
        """Returns the spline order"""
        return self._order

    @property
    def pieces(self) -> int:
        """Returns the number of the spline pieces"""
        return self._pieces

    @property
    def ndim(self) -> int:
        """Returns dimensionality (>1 for multivariate data)"""
        return self._ndim

    def evaluate(self, xi: np.ndarray) -> np.ndarray:
        """Evaluates the spline for the given data point(s)"""

        # For each data site, compute its break interval
        mesh = self.breaks[1:-1]
        edges = np.hstack((-np.inf, mesh, np.inf))

        index = np.digitize(xi, edges)

        nanx = np.flatnonzero(index == 0)
        index = np.fmin(index, mesh.size + 1)
        index[nanx] = 1
        index -= 1

        # Go to local coordinates
        xi = xi - self.breaks[index]

        # Apply nested multiplication
        values = self.coeffs[:, index]

        for i in range(1, self.order):
            index += self.pieces
            values = xi * values + self.coeffs[:, index]

        return values


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

    axis : int
        Axis along which ``ydata`` is assumed to be varying.
        Meaning that for x[i] the corresponding values are np.take(ydata, i, axis=axis).
        By default is -1 (the last axis).
    """

    def __init__(self,
                 xdata: UnivariateDataType,
                 ydata: MultivariateDataType,
                 weights: ty.Optional[UnivariateDataType] = None,
                 smooth: ty.Optional[float] = None,
                 axis: int = -1):

        (self._xdata,
         self._ydata,
         self._weights,
         self._shape) = self._prepare_data(xdata, ydata, weights, axis)

        self._ydim = self._ydata.shape[0]
        self._axis = axis

        self._spline, self._smooth = self._make_spline(smooth)

    def __call__(self, xi: UnivariateDataType) -> np.ndarray:
        """Evaluate the spline for given data
        """
        xi = ty.cast(np.ndarray, np.asarray(xi, dtype=np.float64))

        if xi.ndim > 1:  # pragma: no cover
            raise ValueError("'xi' data must be a 1-d array.")

        yi = self._spline.evaluate(xi)

        shape = list(self._shape)
        shape[self._axis] = xi.size
        shape = tuple(shape)

        if yi.shape != shape:
            # Reshape values 2-D NxM array to N-D array with original shape
            yi = from_2d(yi, shape, self._axis)

        return yi

    @property
    def smooth(self) -> float:
        """Returns the smoothing parameter

        Returns
        -------
        smooth : float
            Smooth factor in the range [0, 1]
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

        yshape = list(ydata.shape)

        if yshape[axis] != xdata.size:
            raise ValueError(
                f"'ydata' data must be a 1-D or N-D array with shape[{axis}] "
                f"that is equal to 'xdata' size ({xdata.size})")

        # Reshape ydata N-D array to 2-D NxM array where N is the data
        # dimension and M is the number of data points.
        ydata = to_2d(ydata, axis)

        if weights is None:
            weights = np.ones_like(xdata)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.size != xdata.size:
                raise ValueError('Weights vector size must be equal of xdata size')

        return xdata, ydata, weights, yshape

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

    def _make_spline(self, smooth: ty.Optional[float]) -> ty.Tuple[SplinePPForm, float]:
        pcount = self._xdata.size
        dx = np.diff(self._xdata)

        if not all(dx > 0):  # pragma: no cover
            raise ValueError(
                "Items of 'xdata' vector must satisfy the condition: x1 < x2 < ... < xN")

        dy = np.diff(self._ydata, axis=1)
        dy_dx = dy / dx

        if pcount == 2:
            # The corner case for the data with 2 points (1 breaks interval)
            # In this case we have 2-ordered spline and linear interpolation in fact
            yi = self._ydata[:, 0][:, np.newaxis]
            coeffs = np.hstack((dy_dx, yi))

            spline = SplinePPForm(breaks=self._xdata, coeffs=coeffs)
            p = 1.

            return spline, p

        # Create diagonal sparse matrices
        diags_r = np.vstack((dx[1:], 2 * (dx[1:] + dx[:-1]), dx[:-1]))
        r = sp.spdiags(diags_r, [-1, 0, 1], pcount - 2, pcount - 2)

        dx_recip = 1. / dx
        diags_qtw = np.vstack((dx_recip[:-1], -(dx_recip[1:] + dx_recip[:-1]), dx_recip[1:]))
        diags_sqrw_recip = 1. / np.sqrt(self._weights)

        qtw = (sp.diags(diags_qtw, [0, 1, 2], (pcount - 2, pcount)) @
               sp.diags(diags_sqrw_recip, 0, (pcount, pcount)))
        qtw = qtw @ qtw.T

        if smooth is None:
            p = self._compute_smooth(r, qtw)
        else:
            p = smooth

        pp = (6. * (1. - p))

        # Solve linear system for the 2nd derivatives
        a = pp * qtw + p * r
        b = np.diff(dy_dx, axis=1).T

        u = la.spsolve(a, b)
        if u.ndim < 2:
            u = u[np.newaxis]
        if self._ydim == 1:
            u = u.T

        dx = dx[:, np.newaxis]

        vpad = functools.partial(np.pad, pad_width=[(1, 1), (0, 0)], mode='constant')

        d1 = np.diff(vpad(u), axis=0) / dx
        d2 = np.diff(vpad(d1), axis=0)

        diags_w_recip = 1. / self._weights
        w = sp.diags(diags_w_recip, 0, (pcount, pcount))

        yi = self._ydata.T - (pp * w) @ d2
        pu = vpad(p * u)

        p1 = np.diff(pu, axis=0) / dx
        p2 = 3. * pu[:-1, :]
        p3 = np.diff(yi, axis=0) / dx - dx * (2. * pu[:-1, :] + pu[1:, :])
        p4 = yi[:-1, :]

        coeffs = np.vstack((p1, p2, p3, p4)).T
        spline = SplinePPForm(breaks=self._xdata, coeffs=coeffs)

        return spline, p
