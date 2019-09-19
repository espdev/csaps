# -*- coding: utf-8 -*-

import abc
import typing as ty
import collections.abc as c_abc

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la

from csaps._types import (
    UnivariateDataType,
    UnivariateVectorizedDataType,
    MultivariateDataType,
    NdGridDataType,
)

from csaps._utils import to_2d, from_2d


TData = ty.TypeVar('TData', np.ndarray, ty.Sequence[np.ndarray])
TProps = ty.TypeVar('TProps', int, ty.Tuple[int, ...])


class SplinePPFormBase(abc.ABC, ty.Generic[TData, TProps]):

    @property
    @abc.abstractmethod
    def breaks(self) -> TData:
        """Returns breaks data

        Returns
        -------
        breaks : Union[np.ndarray, ty.Tuple[np.ndarray, ...]]
            Breaks data
        """

    @property
    @abc.abstractmethod
    def coeffs(self) -> np.ndarray:
        """Returns a spline coefficients 2-D array

        Returns
        -------
        coeffs : np.ndarray
            Coefficients 2-D array
        """

    @property
    @abc.abstractmethod
    def pieces(self) -> TProps:
        """Returns a spline pieces data

        Returns
        -------
        pieces : ty.Union[int, ty.Tuple[int, ...]]
            Returns a spline pieces data
        """

    @property
    @abc.abstractmethod
    def order(self) -> TProps:
        """Returns a spline order

        Returns
        -------
        order : ty.Union[int, ty.Tuple[int, ...]]
            Returns a spline order
        """

    @property
    @abc.abstractmethod
    def ndim(self) -> int:
        """Returns a spline dimension count

        Returns
        -------
        ndim : int
            A spline dimension count
        """

    @abc.abstractmethod
    def evaluate(self, xi: TData) -> np.ndarray:
        """Evaluates the spline for given data sites

        Parameters
        ----------
        xi : UnivariateDataType, NdGridDataType
            X data vector or list of vectors for multivariate spline

        Returns
        -------
        data : np.ndarray
            Interpolated/smoothed data
        """

    def __repr__(self):
        return (
            '{}\n'
            '  breaks: {}\n'
            '  coeffs: {} shape\n'
            '  pieces: {}\n'
            '  order: {}\n'
            '  ndim: {}\n'
        ).format(
            type(self).__name__, self.breaks, self.coeffs.shape,
            self.pieces, self.order, self.ndim)


class SplinePPForm(SplinePPFormBase[np.ndarray, int]):
    """Univariate/multivariate spline representation in PP-form

    Parameters
    ----------
    breaks : np.ndarray
        Breaks values 1-d array
    coeffs : np.ndarray
        Spline coefficients 2-d array
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


class NdGridSplinePPForm(SplinePPFormBase[ty.Sequence[np.ndarray], ty.Tuple[int, ...]]):
    """N-D grid spline representation in PP-form

    Parameters
    ----------
    breaks : np.ndarray
        Breaks values 1-d array
    coeffs : np.ndarray
        Spline coefficients 2-d array
    """

    def __init__(self, breaks: ty.Sequence[np.ndarray], coeffs: np.ndarray) -> None:
        self._breaks = breaks
        self._coeffs = coeffs
        self._pieces = tuple(x.size - 1 for x in breaks)
        self._order = tuple(s // p for s, p in zip(coeffs.shape[1:], self._pieces))
        self._ndim = len(breaks)

    @property
    def breaks(self) -> ty.Sequence[np.ndarray]:
        return self._breaks

    @property
    def coeffs(self) -> np.ndarray:
        return self._coeffs

    @property
    def pieces(self) -> ty.Tuple[int, ...]:
        return self._pieces

    @property
    def order(self) -> ty.Tuple[int, ...]:
        return self._order

    @property
    def ndim(self) -> int:
        return self._ndim

    def evaluate(self, xi: ty.Sequence[np.ndarray]) -> np.ndarray:
        yi = self.coeffs.copy()
        sizey = list(yi.shape)
        nsize = tuple(x.size for x in xi)

        for i in range(self.ndim - 1, -1, -1):
            ndim = int(np.prod(sizey[:self.ndim]))
            coeffs = yi.reshape((ndim * self.pieces[i], self.order[i]), order='F')

            spp = SplinePPForm(self.breaks[i], coeffs, ndim=ndim, shape=(ndim, xi[i].size))
            yi = spp.evaluate(xi[i])

            yi = yi.reshape((*sizey[:self.ndim], nsize[i]), order='F')
            axes = (0, self.ndim, *range(1, self.ndim))
            yi = yi.transpose(axes)
            sizey = list(yi.shape)

        return yi.reshape(nsize, order='F')


TSmooth = ty.TypeVar('TSmooth', float, ty.Tuple[float, ...])
TSpline = ty.TypeVar('TSpline', SplinePPForm, NdGridSplinePPForm)
TXi = ty.TypeVar('TXi', UnivariateDataType, NdGridDataType)


class ISmoothingSpline(abc.ABC, ty.Generic[TSmooth, TSpline, TXi]):
    """The interface class for smooting splines
    """

    @property
    @abc.abstractmethod
    def smooth(self) -> TSmooth:
        """Returns smoothing parameter
        """

    @property
    @abc.abstractmethod
    def spline(self) -> TSpline:
        """Returns spline representation in PP-form
        """

    @abc.abstractmethod
    def __call__(self, xi: TXi) -> np.ndarray:
        """Evaluates spline on the data sites
        """


class UnivariateCubicSmoothingSpline(ISmoothingSpline[float, SplinePPForm, UnivariateDataType]):
    """Univariate cubic smoothing spline

    Parameters
    ----------
    xdata : np.ndarray, list
        X input 1D data vector (data sites: x1 < x2 < ... < xN)
    ydata : np.ndarray, list
        Y input 1D data vector or ND-array with shape[axis] equal of X data size)
    weights : np.ndarray, list
        [Optional] Weights 1D vector with size equal of xdata size
    smooth : float
        [Optional] Smoothing parameter in range [0, 1] where:
            - 0: The smoothing spline is the least-squares straight line fit
            - 1: The cubic spline interpolant with natural condition
    axis : int
        Axis along which "ydata" is assumed to be varying.
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
        """Evaluate the spline's approximation for given data
        """
        xi = ty.cast(np.ndarray, np.asarray(xi, dtype=np.float64))

        if xi.ndim > 1:
            raise ValueError('"xi" data must be a 1-d array.')

        return self._spline.evaluate(xi)

    @property
    def smooth(self) -> float:
        """Returns smooth factor

        Returns
        -------
        smooth : float
            Smooth factor in the range [0, 1]
        """
        return self._smooth

    @property
    def spline(self) -> SplinePPForm:
        """Returns the spline description in 'SplinePPForm' instance

        Returns
        -------
        spline : SplinePPForm
            The spline description in 'SplinePPForm' instance
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
                '"ydata" data must be a 1-D or N-D array with shape[{}] that is equal to "xdata" size ({})'.format(
                    axis, xdata.size))

        # Reshape ydata N-D array to 2-D NxM array where N is the data
        # dimension and M is the number of data points.
        ydata = to_2d(ydata, axis)

        if weights is None:
            weights = np.ones_like(xdata)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.size != xdata.size:
                raise ValueError(
                    'Weights vector size must be equal of xdata size')

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

        if not all(dx > 0):
            raise ValueError(
                'Items of xdata vector must satisfy the condition: x1 < x2 < ... < xN')

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

            coeffs = np.hstack((
                (np.diff(c3, axis=0) / dx).T,
                3. * c3[:-1, :].T,
                c2.T,
                yi[:-1, :].T
            ))

            cf_shape = ((pcount - 1) * self._ydim, 4)
            coeffs = coeffs.reshape(cf_shape, order='F')
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


class MultivariateCubicSmoothingSpline(ISmoothingSpline[float, SplinePPForm, UnivariateDataType]):
    """Multivariate parametrized cubic smoothing spline

    Class implments multivariate data approximation via cubic smoothing spline with
    parametric data sites vector `t`: `X(t), Y(t), ..., M(t)`.

    This approach with parametrization allows us to use univariate splines for
    approximation multivariate data.

    For example:

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
    tdata : np.ndarray, list
        [Optional] Parametric vector of data sites with condition: `t1 < t2 < ... < tN`.
        If it is not set will be computed automatically.
    weights : np.ndarray, list
        [Optional] Weights 1D vector with size equal of N
    smooth : float
        [Optional] Smoothing parameter in range [0, 1] where:
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

        ydata = ty.cast(np.ndarray, np.asarray(ydata, dtype=np.float64))

        if tdata is None:
            tdata = self._compute_tdata(to_2d(ydata, axis))

        tdata = ty.cast(np.ndarray, np.asarray(tdata, dtype=np.float64))

        if tdata.size != ydata.shape[-1]:
            raise ValueError('"tdata" size must be equal to "ydata" shape[{}] size ({})'.format(
                axis, ydata.shape[axis]))

        self._tdata = tdata

        # Use vectorization for compute spline for every dimension from t
        self._univariate_spline = UnivariateCubicSmoothingSpline(
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
        """Returns smooth factor

        Returns
        -------
        smooth : float
            Smooth factor in the range [0, 1]
        """
        return self._univariate_spline.smooth

    @property
    def spline(self) -> SplinePPForm:
        """Returns the spline description in 'SplinePPForm' instance

        Returns
        -------
        spline : SplinePPForm
            The spline description in 'SplinePPForm' instance
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
        tail = np.sqrt(np.sum(np.diff(data, axis=1)**2, axis=0))
        return np.cumsum(np.hstack((head, tail)))


class NdGridCubicSmoothingSpline(ISmoothingSpline[ty.Tuple[float, ...], NdGridSplinePPForm, NdGridDataType]):
    """ND-Gridded cubic smoothing spline

    Class implments ND-gridded data approximation via cubic smoothing spline
    (piecewise tensor product polynomial).

    Parameters
    ----------
    xdata : list, tuple
        X data site vectors for each dimensions. These vectors determine ND-grid.
        For example::

            # 2D grid
            x = [np.linspace(0, 5, 21), np.linspace(0, 6, 25)]

    ydata : np.ndarray
        Y input data ND-array with shape equal X data vector sizes
    weights : list, tuple
        [Optional] Weights data vectors for all dimensions with size equal xdata sizes
    smooth : float
        [Optional] Smoothing parameter (or list of parameters for each dimension) in range [0, 1] where:
            - 0: The smoothing spline is the least-squares straight line fit
            - 1: The cubic spline interpolant with natural condition
    """

    def __init__(self,
                 xdata: NdGridDataType,
                 ydata: np.ndarray,
                 weights: ty.Optional[ty.Union[UnivariateDataType, NdGridDataType]] = None,
                 smooth: ty.Optional[ty.Union[float, ty.Sequence[ty.Optional[float]]]] = None) -> None:

        (self._xdata,
         self._ydata,
         self._weights,
         _smooth) = self._prepare_data(xdata, ydata, weights, smooth)

        self._ndim = len(self._xdata)
        self._spline, self._smooth = self._make_spline(_smooth)

    @property
    def smooth(self) -> ty.Tuple[float, ...]:
        """Returns smooth factor for every axis

        Returns
        -------
        smooth : Tuple[float, ...]
            Smooth factor in the range [0, 1] for every axis
        """
        return self._smooth

    @property
    def spline(self) -> NdGridSplinePPForm:
        """Returns the spline description in 'NdGridSplinePPForm' instance

        Returns
        -------
        spline : SplinePPForm
            The spline description in 'SplinePPForm' instance
        """
        return self._spline

    @staticmethod
    def _prepare_grid_vectors(data, name) -> ty.Tuple[np.ndarray, ...]:
        if not isinstance(data, c_abc.Sequence):
            raise TypeError('{} must be sequence of vectors'.format(name))

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
        xdata = cls._prepare_grid_vectors(xdata, 'xdata')
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
            weights = cls._prepare_grid_vectors(weights, 'weights')

        if len(weights) != data_ndim:
            raise ValueError(
                'weights ({}) and xdata ({}) dimensions mismatch'.format(len(weights), data_ndim))

        for w, x in zip(weights, xdata):
            if w is not None:
                if w.size != x.size:
                    raise ValueError(
                        'weights ({}) and xdata ({}) dimension size mismatch'.format(w, x))

        if not smooth:
            smooth = [None] * data_ndim

        if not isinstance(smooth, c_abc.Sequence):
            smooth = [float(smooth)] * data_ndim
        else:
            smooth = list(smooth)

        if len(smooth) != data_ndim:
            raise ValueError(
                'Number of smoothing parameter values must be equal '
                'number of dimensions ({})'.format(data_ndim))

        return xdata, ydata, weights, smooth

    def __call__(self, xi: NdGridDataType) -> np.ndarray:
        xi = self._prepare_grid_vectors(xi, 'xi')

        if len(xi) != self._ndim:
            raise ValueError(
                'xi ({}) and xdata ({}) dimensions mismatch'.format(len(xi), self._ndim))

        return self._spline.evaluate(xi)

    def _make_spline(self, smooth: ty.List[ty.Optional[float]]) -> ty.Tuple[NdGridSplinePPForm, ty.Tuple[float, ...]]:
        sizey = [1] + list(self._ydata.shape)
        ydata = self._ydata.reshape(sizey, order='F').copy()
        _smooth = []

        # Perform coordinatewise smoothing spline computing
        for i in range(self._ndim-1, -1, -1):
            shape_i = (np.prod(sizey[:-1]), sizey[-1])
            ydata_i = ydata.reshape(shape_i, order='F')

            s = UnivariateCubicSmoothingSpline(
                self._xdata[i], ydata_i, self._weights[i], smooth[i])

            _smooth.append(s.smooth)
            sizey[-1] = s.spline.pieces * s.spline.order
            ydata = s.spline.coeffs.reshape(sizey, order='F')

            if self._ndim > 1:
                axes = (0, self._ndim, *range(1, self._ndim))
                ydata = ydata.transpose(axes)
                sizey = list(ydata.shape)

        return NdGridSplinePPForm(self._xdata, ydata), tuple(_smooth)
