# -*- coding: utf-8 -*-

"""
Univariate/multivariate cubic smoothing spline implementation

"""

import typing as ty
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la

from ._base import SplinePPFormBase, ISmoothingSpline
from ._types import UnivariateDataType, UnivariateVectorizedDataType, MultivariateDataType
from ._reshape import from_2d, to_2d


class SplinePPForm(SplinePPFormBase[np.ndarray, int]):
    """Univariate/multivariate spline representation in PP-form

    Parameters
    ----------
    breaks : np.ndarray
        Breaks values 1-D array

    coeffs : np.ndarray
        Spline coefficients 2-D array

    ndim : int
        Spline dimension

    shape : Sequence[int]
        It determines Y data shape

    axis : int
        Axis along which values are assumed to be varying
    """

    def __init__(self, breaks: np.ndarray, coeffs: np.ndarray, ndim: int,
                 shape: ty.Sequence[int], axis: int = -1) -> None:
        self._breaks = breaks
        self._coeffs = coeffs
        self._pieces = np.prod(coeffs.shape[:-1]) // ndim
        self._order = coeffs.shape[-1]
        self._ndim = ndim

        self._shape = shape
        self._axis = axis

    @property
    def breaks(self) -> np.ndarray:
        return self._breaks

    @property
    def coeffs(self) -> np.ndarray:
        return self._coeffs

    @property
    def pieces(self) -> int:
        return self._pieces

    @property
    def order(self) -> int:
        return self._order

    @property
    def ndim(self) -> int:
        return self._ndim

    def evaluate(self, xi: np.ndarray) -> np.ndarray:
        shape = list(self._shape)
        shape[self._axis] = xi.size

        # For each data site, compute its break interval
        mesh = self.breaks[1:-1]
        edges = np.hstack((-np.inf, mesh, np.inf))

        index = np.digitize(xi, edges)

        nanx = np.flatnonzero(index == 0)
        index = np.fmin(index, mesh.size + 1)
        index[nanx] = 1

        # Go to local coordinates
        xi = xi - self.breaks[index - 1]
        d = self.ndim
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
        values = self._coeffs[index, 0].T

        for i in range(1, self._coeffs.shape[1]):
            values = xi * values + self._coeffs[index, i].T

        values = values.reshape((d, lx), order='F').squeeze()

        if values.shape != shape:
            # Reshape values 2-D NxM array to N-D array with original shape
            values = from_2d(values, shape, self._axis)

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
                 ydata: UnivariateVectorizedDataType,
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
            raise ValueError('"xi" data must be a 1-d array.')

        return self._spline.evaluate(xi)

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
            raise ValueError('xdata must be a vector')
        if xdata.size < 2:
            raise ValueError('xdata must contain at least 2 data points.')

        yshape = list(ydata.shape)

        if yshape[axis] != xdata.size:
            raise ValueError(
                f'"ydata" data must be a 1-D or N-D array with shape[{axis}] '
                f'that is equal to "xdata" size ({xdata.size})')

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

    @profile
    def _make_spline(self, smooth: ty.Optional[float]) -> ty.Tuple[SplinePPForm, float]:
        pcount = self._xdata.size
        dx = np.diff(self._xdata)

        if not all(dx > 0):  # pragma: no cover
            raise ValueError('Items of xdata vector must satisfy the condition: x1 < x2 < ... < xN')

        dy = np.diff(self._ydata, axis=1)
        dy_dx = dy / dx

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

            if smooth is None:
                p = self._compute_smooth(r, qtwq)
            else:
                p = smooth

            a = (6. * (1. - p)) * qtwq + p * r
            b = np.diff(dy_dx, axis=1).T

            u = la.spsolve(a, b)
            if u.ndim < 2:
                u = u[np.newaxis]
            if self._ydim == 1:
                u = u.T

            dx = dx[:, np.newaxis]
            d_pad = np.zeros((1, self._ydim))

            d1 = np.diff(np.vstack((d_pad, u, d_pad)), axis=0) / dx
            d2 = np.diff(np.vstack((d_pad, d1, d_pad)), axis=0)

            yi = self._ydata.T - ((6. * (1. - p)) * w) @ d2
            c3 = np.vstack((d_pad, p * u, d_pad))
            c2 = np.diff(yi, axis=0) / dx - dx * (2. * c3[:-1, :] + c3[1:, :])

            coeffs = np.stack((
                (np.diff(c3, axis=0) / dx).ravel(),
                3. * c3[:-1, :].ravel(),
                c2.ravel(),
                yi[:-1, :].ravel()
            ), axis=1)
        else:
            p = 1.
            yi = self._ydata[:, 0][:, np.newaxis]
            coeffs = np.array(np.hstack((dy_dx, yi)), ndmin=2)

        spline = SplinePPForm(
            breaks=self._xdata,
            coeffs=coeffs,
            ndim=self._ydim,
            shape=self._shape,
            axis=self._axis
        )

        return spline, p


class UnivariateCubicSmoothingSpline(ISmoothingSpline[SplinePPForm, float, UnivariateDataType]):
    __doc__ = CubicSmoothingSpline.__doc__

    def __init__(self,
                 xdata: UnivariateDataType,
                 ydata: UnivariateVectorizedDataType,
                 weights: ty.Optional[UnivariateDataType] = None,
                 smooth: ty.Optional[float] = None,
                 axis: int = -1) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                "'UnivariateCubicSmoothingSpline' class is deprecated "
                "and will be removed in the future version. "
                "Use 'CubicSmoothingSpline' class instead.", stacklevel=2)

        self._cssp = CubicSmoothingSpline(
            xdata, ydata, weights=weights, smooth=smooth, axis=axis)

    @property
    def smooth(self) -> float:
        return self._cssp.smooth

    @property
    def spline(self) -> SplinePPForm:
        return self._cssp.spline

    def __call__(self, xi: UnivariateDataType) -> np.ndarray:
        return self._cssp(xi)


# For case isinstance(CubicSmoothingSpline(...), UnivariateCubicSmoothingSpline)
UnivariateCubicSmoothingSpline.register(CubicSmoothingSpline)


class MultivariateCubicSmoothingSpline(ISmoothingSpline[SplinePPForm, float, UnivariateDataType]):
    """Multivariate parametrized cubic smoothing spline

    Class implments multivariate data approximation via cubic smoothing spline with
    parametric data sites vector `t`: `X(t), Y(t), ..., M(t)`.

    This approach with parametrization allows us to use univariate splines for
    approximation multivariate data.

    For example:

    .. code-block:: python

        # 3D data
        data = [
            # Data vectors   Dimension
            (2, 4, 1, 3),  # X
            (1, 4, 3, 2),  # Y
            (3, 4, 1, 5),  # Z
        ]

        x, y, z = 0, 1, 2

        t = (0, 1, 2, 3)  # parametric vector of data sites (t1 < t2 < ... < tN)

        # Construct multivariate spline from t and X, Y, Z
        sx = UnivariateCubicSmoothingSpline(t, data[x])
        sy = UnivariateCubicSmoothingSpline(t, data[y])
        sz = UnivariateCubicSmoothingSpline(t, data[z])

        # Or the same with using vectorization
        sxyz = UnivariateCubicSmoothingSpline(t, data)

    Parameters
    ----------

    ydata : np.ndarray, array-like
        Input multivariate data vectors. N-D array.

    tdata : [*Optional*] np.ndarray, list
        Parametric vector of data sites with condition: `t1 < t2 < ... < tN`.
        If it is not set will be computed automatically.

    weights : [*Optional*] np.ndarray, list
        Weights 1D vector with size equal of N

    smooth : [*Optional*] float
        Smoothing parameter in range [0, 1] where:
            - 0: The smoothing spline is the least-squares straight line fit
            - 1: The cubic spline interpolant with natural condition

    axis : int
        Axis along which "ydata" is assumed to be varying.
        Meaning that for x[i] the corresponding values are np.take(ydata, i, axis=axis).
        By default is -1 (the last axis).

    See Also
    --------
    UnivariateCubicSmoothingSpline

    """

    def __init__(self,
                 ydata: MultivariateDataType,
                 tdata: ty.Optional[UnivariateDataType] = None,
                 weights: ty.Optional[UnivariateDataType] = None,
                 smooth: ty.Optional[float] = None,
                 axis: int = -1):

        with warnings.catch_warnings():
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                "'MultivariateCubicSmoothingSpline' class is deprecated "
                "and will be removed in the future version. "
                "Use 'CubicSmoothingSpline' class instead.", stacklevel=2)

        ydata = ty.cast(np.ndarray, np.asarray(ydata, dtype=np.float64))

        if tdata is None:
            tdata = self._compute_tdata(to_2d(ydata, axis))

        tdata = ty.cast(np.ndarray, np.asarray(tdata, dtype=np.float64))

        if tdata.size != ydata.shape[-1]:  # pragma: no cover
            raise ValueError(f'"tdata" size must be equal to "ydata" shape[{axis}] size ({ydata.shape[axis]})')

        self._tdata = tdata

        # Use vectorization for compute spline for every dimension from t
        self._univariate_spline = CubicSmoothingSpline(
            xdata=tdata,
            ydata=ydata,
            weights=weights,
            smooth=smooth,
            axis=axis,
        )

    def __call__(self, ti: UnivariateDataType):
        return self._univariate_spline(ti)

    @property
    def smooth(self) -> float:
        """Returns the smoothing parameter

        Returns
        -------
        smooth : float
            Smooth factor in the range [0, 1]
        """
        return self._univariate_spline.smooth

    @property
    def spline(self) -> SplinePPForm:
        """Returns the spline description in `SplinePPForm` instance

        Returns
        -------
        spline : SplinePPForm
            The spline description in :class:`SplinePPForm` instance
        """
        return self._univariate_spline.spline

    @property
    def t(self) -> np.ndarray:
        """Returns parametrization data vector

        Returns
        -------
        t : np.ndarray
            The parametrization data vector
        """
        return self._tdata

    @staticmethod
    def _compute_tdata(data):
        """Computes "natural" t parametrization vector for N-dimensional data

        .. code-block::

            t_1 = 0
            t_i+1 = t_i + sqrt((x_i+1 - x_i)**2 + (y_i+1 - y_i)**2 + ... + (n_i+1 - n_i)**2)
        """
        head = 0.
        tail = np.sqrt(np.sum(np.diff(data, axis=1) ** 2, axis=0))
        return np.cumsum(np.hstack((head, tail)))
