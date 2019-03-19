# -*- coding: utf-8 -*-

"""
Cubic spline approximation (smoothing)

"""

import typing as t

import numpy as np

import scipy.sparse as sp
import scipy.sparse.linalg as la


__version__ = '0.2.0'


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
        [Optional] Weights 1D vector with size equal of xdata size
    smooth : float
        [Optional] Smoothing parameter in range [0, 1] where:
            - 0: The smoothing spline is the least-squares straight line fit
            - 1: The cubic spline interpolant
    """

    def __init__(self, xdata: DataType, ydata: DataType,
                 weights: t.Optional[DataType] = None,
                 smooth: t.Optional[float] = None):
        self._coeffs = None
        self._pieces = 0
        self._smooth = smooth

        (self._xdata,
         self._ydata,
         self._weights,
         self._data_shape) = self._prepare_data(xdata, ydata, weights)

        self._yd = self._ydata.shape[0]
        self._axis = self._ydata.ndim - 1

        self._make_spline()

    def __call__(self, xi: DataType):
        """Evaluate the spline's approximation for given data
        """
        xi = np.asarray(xi, dtype=np.float64)

        if xi.ndim > 1:
            raise ValueError('XI data must be a vector.')

        self._data_shape[-1] = xi.size

        return self._evaluate(xi)

    @property
    def smooth(self):
        return self._smooth

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

        data_shape = list(ydata.shape)

        if xdata.ndim > 1:
            raise ValueError('xdata must be a vector')
        if xdata.size < 2:
            raise ValueError('xdata must contain at least 2 data points.')

        if ydata.ndim > 1:
            if data_shape[-1] != xdata.size:
                raise ValueError(
                    'ydata data must be a vector or '
                    'ND-array with shape[-1] equal of xdata.size')

            if ydata.ndim > 2:
                ydata = ydata.reshape((np.prod(data_shape[:-1]), data_shape[-1]))
        else:
            if ydata.size != xdata.size:
                raise ValueError('ydata vector size must be equal of xdata size')

            ydata = np.array(ydata, ndmin=2)

        if weights is None:
            weights = np.ones_like(xdata)
        else:
            weights = np.asarray(weights, dtype=np.float64)

            if weights.size != xdata.size:
                raise ValueError(
                    'Weights vector size must be equal of xdata size')

        return xdata, ydata, weights, data_shape

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
        dy = np.diff(self._ydata, axis=self._axis)
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
            b = np.diff(divdydx, axis=self._axis).T
            u = np.array(la.spsolve(a, b), ndmin=2)

            if self._yd == 1:
                u = u.T

            dx = np.array(dx, ndmin=2).T
            d_pad = np.zeros((1, self._yd))

            d1 = np.diff(np.vstack((d_pad, u, d_pad)), axis=0) / dx
            d2 = np.diff(np.vstack((d_pad, d1, d_pad)), axis=0)

            yi = np.array(self._ydata, ndmin=2).T
            yi = yi - ((6. * (1. - p)) * w) @ d2
            c3 = np.vstack((d_pad, p * u, d_pad))
            c2 = np.diff(yi, axis=0) / dx - dx * (2. * c3[:-1, :] + c3[1:, :])

            coeffs = np.hstack((
                (np.diff(c3, axis=0) / dx).T,
                3. * c3[:-1, :].T,
                c2.T,
                yi[:-1, :].T
            ))

            c_shape = ((pcount - 1) * self._yd, 4)
            coeffs = coeffs.reshape(c_shape, order='F')
        else:
            p = 1.
            coeffs = np.array(np.hstack(
                (divdydx, np.array(self._ydata[:, 0], ndmin=2).T)), ndmin=2)

        self._smooth = p
        self._breaks = self._xdata.copy()
        self._coeffs = coeffs
        self._pieces = np.prod(coeffs.shape[:-1]) // self._yd

    def _evaluate(self, xi):
        # For each data site, compute its break interval
        mesh = self._xdata[1:-1]
        edges = np.hstack((-np.inf, mesh, np.inf))

        index = np.digitize(xi, edges)

        nanx = np.flatnonzero(index == 0)
        index = np.fmin(index, mesh.size + 1)
        index[nanx] = 1

        # Go to local coordinates
        xi = xi - self._xdata[index-1]
        d = self._yd
        lx = len(xi)

        if d > 1:
            xi_shape = (1, d * lx)
            xi_ndm = np.array(xi, ndmin=2)
            xi = np.reshape(np.repeat(xi_ndm, d, axis=0), xi_shape, order='F')

            index_rep = (np.repeat(np.array(1 + d * index, ndmin=2), d, axis=0)
                         + np.repeat(np.array(np.r_[-d:0], ndmin=2).T, lx, axis=1))
            index = np.reshape(index_rep, (d * lx, 1), order='F')

        index -= 1

        # Apply nested multiplication
        values = self.coeffs[index, 0].T

        for i in range(1, self.coeffs.shape[1]):
            values = xi * values + self.coeffs[index, i].T

        values = values.reshape((d, lx), order='F').squeeze()

        if values.shape != self._data_shape:
            values = values.reshape(self._data_shape)

        return values
