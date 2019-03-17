# -*- coding: utf-8 -*-

"""
Cubic spline approximation (smoothing)

"""

import typing as t

import numpy as np

import scipy.sparse as sp
import scipy.sparse.linalg as la


__version__ = '0.1.0'


DataType = t.Union[t.Sequence, np.ndarray]


class UnivariateCubicSmoothingSpline:
    """Univariate cubic smoothing spline

    Parameters
    ----------
    xdata : np.ndarray, list
        X input 1D data vector
    ydata : np.ndarray, list
        Y input 1D data vector or ND-array with shape[-1] equal of X data size)
    weights : np.ndarray, list
        Weights 1D vector or ND-array with shape equal of Y data shape
    smooth : float
        Smoothing parameter in range [0, 1] where:
            - 0: The smoothing spline is the least-squares straight line fit
            - 1: The cubic spline interpolant
    """

    def __init__(self, xdata: DataType, ydata: DataType,
                 weights: t.Optional[DataType] = None,
                 smooth: t.Optional[float] = None):
        self._breaks = None
        self._coeffs = None
        self._pieces = 0
        self._smooth = smooth

        (self._xdata,
         self._ydata,
         self._weights) = self._prepare_data(xdata, ydata, weights)

        self._make_spline()

    def __call__(self, xi: DataType):
        """Evaluate the spline's approximation for given data
        """
        xi = np.asarray(xi, dtype=np.float64)

        if xi.ndim > 1:
            raise ValueError('XI data must be a vector.')

        return self._evaluate(xi)

    @property
    def smooth(self):
        return self._smooth

    @property
    def breaks(self):
        return self._breaks

    @property
    def coeffs(self):
        return self._coeffs

    @property
    def pieces(self):
        return self._pieces

    @staticmethod
    def _prepare_data(xdata, ydata, weights):
        xdata = np.asarray(xdata, dtype=np.float64)
        ydata = np.asarray(ydata, dtype=np.float64)

        if xdata.ndim > 1:
            raise ValueError('xdata must be a vector')
        if xdata.size < 2:
            raise ValueError('xdata must contain at least 2 data points.')

        if ydata.ndim > 1:
            if ydata.shape[-1] != xdata.size:
                raise ValueError(
                    'ydata data must be a vector or '
                    'ND-array with shape[-1] equal of xdata.size')
        else:
            if ydata.size != xdata.size:
                raise ValueError('ydata vector size must be equal of xdata size')

        if weights is None:
            weights = np.ones_like(ydata)
        else:
            weights = np.asarray(weights, dtype=np.float64)

            if weights.ndim > 1:
                if weights.shape != ydata.shape:
                    raise ValueError(
                        'Weights data must be a vector or '
                        'ND-array with shape equal of ydata.shape')
            else:
                if weights.size != xdata.size:
                    raise ValueError(
                        'Weights vector size must be equal of xdata size')

            if ydata.ndim > 1 and weights.ndim == 1:
                weights = np.array(weights, ndmin=ydata.ndim)
                weights = np.ones_like(ydata) * weights

        return xdata, ydata, weights

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

    def _make_spline(self):
        pcount = self._xdata.size

        dx = np.diff(self._xdata)
        dy = np.diff(self._ydata)
        divdydx = dy / dx

        if pcount > 2:
            # Create diagonal sparse matrices
            diags_r = np.vstack((dx[1:], 2 * (dx[1:] + dx[:-1]), dx[:-1]))
            r = sp.spdiags(diags_r, [-1, 0, 1], pcount - 2, pcount - 2)

            odx = 1. / dx
            diags_qt = np.vstack((odx[:-1], -(odx[1:] + odx[:-1]), odx[1:]))
            qt = sp.diags(diags_qt, [0, 1, 2], (pcount - 2, pcount))

            ow = 1. / self._weights
            osqw = 1. / np.sqrt(self._weights)  # type: np.ndarray
            w = sp.diags(ow, 0, (pcount, pcount))
            qtw = qt @ sp.diags(osqw, 0, (pcount, pcount))

            # Solve linear system for the 2nd derivatives
            qtwq = qtw @ qtw.T

            if self._smooth:
                p = self._smooth
            else:
                p = self._compute_smooth(r, qtwq)

            a = (6. * (1. - p)) * qtwq + p * r
            b = np.diff(divdydx)
            u = la.spsolve(a, b)

            d1 = np.diff(np.hstack((0., u, 0.))) / dx
            d2 = np.diff(np.hstack((0., d1, 0.)))

            yi = self._ydata - ((6. * (1. - p)) * w) @ d2
            c3 = np.hstack((0., p * u, 0.))
            c2 = np.diff(yi) / dx - dx * (2. * c3[:-1] + c3[1:])

            coeffs = np.hstack((np.diff(c3) / dx, 3. * c3[:-1], c2, yi[:-1]))
            coeffs = coeffs.reshape((pcount - 1, 4), order='F')
        else:
            p = 1.
            coeffs = np.array(np.hstack((divdydx, self._ydata[0])), ndmin=2)

        self._smooth = p
        self._breaks = self._xdata.copy()
        self._coeffs = coeffs
        self._pieces = coeffs.shape[0]

    def _evaluate(self, xi):
        # For each data site, compute its break interval
        mesh = self._breaks[1:-1]
        edges = np.hstack((-np.inf, mesh, np.inf))

        index = np.digitize(xi, edges)

        nanx = np.flatnonzero(index == 0)
        index = np.fmin(index, mesh.size + 1)
        index[nanx] = 1
        index -= 1

        # Go to local coordinates
        xi = xi - self._breaks[index]

        # Apply nested multiplication
        values = self.coeffs[index, 0]

        for i in range(1, self.coeffs.shape[1]):
            values = xi * values + self.coeffs[index, i]

        return values
