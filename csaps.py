# -*- coding: utf-8 -*-

"""
Cubic spline approximation (smoothing)

"""

import typing as t

import numpy as np

import scipy.sparse as sp
import scipy.sparse.linalg as la


__version__ = '0.2.0'


_UnivariateDataType = t.Union[t.Sequence[t.Union[int, float]], np.ndarray]

_UnivariateVectorizedDataType = t.Union[
    t.Sequence[t.Union[int, float]],
    np.ndarray,
    t.List['_UnivariateVectorizedDataType']
]

_MultivariateDataType = t.Tuple[_UnivariateDataType, ...]


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

    def __init__(self,
                 xdata: _UnivariateDataType,
                 ydata: _UnivariateVectorizedDataType,
                 weights: t.Optional[_UnivariateDataType] = None,
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

    def __call__(self, xi: _UnivariateDataType) -> np.ndarray:
        """Evaluate the spline's approximation for given data
        """
        xi = np.asarray(xi, dtype=np.float64)

        if xi.ndim > 1:
            raise ValueError('XI data must be a vector.')

        self._data_shape[-1] = xi.size

        return self._evaluate(xi)

    @property
    def smooth(self) -> float:
        return self._smooth

    @property
    def breaks(self) -> np.ndarray:
        return self._xdata

    @property
    def coeffs(self) -> np.ndarray:
        return self._coeffs

    @property
    def pieces(self) -> int:
        return self._pieces

    @property
    def order(self) -> int:
        return self._coeffs.shape[-1]

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


class MultivariateCubicSmoothingSpline:
    """Multivariate cubic smoothing spline

    Class implments multivariate (ND-gridded) approximation via cubic
    smoothing spline.

    Parameters
    ----------
    xdata : list, tuple
        X data site vectors for all dimensions (determines a grid). For example::

            # 2D grid
            x = [np.linspace(0, 10, 100), np.linspace(0, 10, 100)]

    ydata : np.ndarray
        Y input ND data array with shape equal X data vector sizes
    weights : list, tuple
        [Optional] Weights data vectors for all dimensions with size equal xdata sizes
    smooth : float
        [Optional] Smoothing parameter (or list of parameters for each dimension) in range [0, 1] where:
            - 0: The smoothing spline is the least-squares straight line fit
            - 1: The cubic spline interpolant
    """

    def __init__(self,
                 xdata: _MultivariateDataType,
                 ydata: np.ndarray,
                 weights: t.Optional[t.Union[_UnivariateDataType,
                                             _MultivariateDataType]] = None,
                 smooth: t.Optional[t.Union[float, t.Sequence[float]]] = None):
        (self._xdata,
         self._ydata,
         self._weights,
         self._smooth) = self._prepare_data(xdata, ydata, weights, smooth)

        self._ndim = len(self._xdata)

        self._coeffs = None
        self._pieces = None
        self._order = None

        self._make_spline()

    @property
    def smooth(self) -> t.Tuple[float, ...]:
        return self._smooth

    @property
    def breaks(self) -> t.Tuple[np.ndarray, ...]:
        return self._xdata

    @property
    def coeffs(self) -> np.ndarray:
        return self._coeffs

    @property
    def pieces(self) -> t.Tuple[int, ...]:
        return self._pieces

    @property
    def order(self) -> t.Tuple[int, ...]:
        return self._order

    @staticmethod
    def _prepare_univariate(data, name):
        if not isinstance(data, (tuple, list)):
            raise TypeError('{} must be list/tuple of vectors'.format(name))

        data = list(data)

        for i, di in enumerate(data):
            di = np.array(di, dtype=np.float64)
            if di.ndim > 1:
                raise ValueError('All {} elements must be vector'.format(name))
            if di.size < 2:
                raise ValueError(
                    '{} must contain at least 2 data points'.format(name))
            data[i] = di

        return tuple(data)

    @classmethod
    def _prepare_data(cls, xdata, ydata, weights, smooth):
        xdata = cls._prepare_univariate(xdata, 'xdata')
        data_ndim = len(xdata)

        if ydata.ndim != data_ndim:
            raise ValueError(
                'ydata must have dimension {} according to xdata'.format(data_ndim))

        for yd, xs in zip(ydata.shape, map(len, xdata)):
            if yd != xs:
                raise ValueError(
                    'ydata ({}) and xdata ({}) dimension size mismatch'.format(yd, xs))

        if not weights:
            weights = [None] * data_ndim
        else:
            weights = cls._prepare_univariate(weights, 'weights')

        if len(weights) != data_ndim:
            raise ValueError(
                'weights ({}) and xdata ({}) dimensions mismatch'.format(
                    len(weights), data_ndim))

        for w, x in zip(weights, xdata):
            if w:
                if w.size != x.size:
                    raise ValueError(
                        'weights ({}) and xdata ({}) dimension size mismatch'.format(w, x))

        if not smooth:
            smooth = [None] * data_ndim

        if not isinstance(smooth, (list, tuple)):
            smooth = [smooth] * data_ndim

        if len(smooth) != data_ndim:
            raise ValueError(
                'Number of smoothing parameter values must be equal '
                'number of dimensions ({})'.format(data_ndim))

        return xdata, ydata, weights, smooth

    def __call__(self, xi: _MultivariateDataType) -> np.ndarray:
        xi = self._prepare_univariate(xi, 'xi')

        if len(xi) != self._ndim:
            raise ValueError(
                'xi ({}) and xdata ({}) dimensions mismatch'.format(
                    len(xi), self._ndim))

        return self._evaluate(xi)

    def _make_spline(self):
        sizey = [1] + list(self._ydata.shape)
        ydata = self._ydata.reshape(sizey, order='F').copy()

        # Perform coordinatewise smoothing spline computing
        for i in range(self._ndim-1, -1, -1):
            shape_i = (np.prod(sizey[:-1]), sizey[-1])
            ydata_i = ydata.reshape(shape_i, order='F')

            spline = UnivariateCubicSmoothingSpline(
                self._xdata[i], ydata_i, self._weights[i], self._smooth[i])

            self._smooth[i] = spline.smooth

            sizey[-1] = spline.pieces * spline.order
            ydata = spline.coeffs.reshape(sizey, order='F')

            if self._ndim > 1:
                axes = (0, self._ndim, *np.r_[1:self._ndim].tolist())
                ydata = ydata.transpose(axes)
                sizey = list(ydata.shape)

        self._coeffs = ydata
        self._pieces = tuple(x.size - 1 for x in self._xdata)
        self._order = tuple((np.array(sizey[1:]) // np.array(self._pieces)).tolist())

    def _evaluate(self, xi):
        raise NotImplementedError
