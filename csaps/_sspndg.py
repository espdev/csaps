# -*- coding: utf-8 -*-

"""
ND-Gridded cubic smoothing spline implementation

"""

import collections.abc as c_abc
import typing as ty

import numpy as np

from csaps._base import SplinePPFormBase, ISmoothingSpline
from csaps._types import UnivariateDataType, NdGridDataType
from csaps._sspumv import SplinePPForm, UnivariateCubicSmoothingSpline


def ndgrid_prepare_data_sites(data, name) -> ty.Tuple[np.ndarray, ...]:
    if not isinstance(data, c_abc.Sequence):
        raise TypeError("'{}' must be a sequence of the vectors.".format(name))

    data = list(data)

    for i, di in enumerate(data):
        di = np.array(di, dtype=np.float64)
        if di.ndim > 1:
            raise ValueError("All '{}' elements must be a vector.".format(name))
        if di.size < 2:
            raise ValueError(
                "'{}' must contain at least 2 data points.".format(name))
        data[i] = di

    return tuple(data)


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


class NdGridCubicSmoothingSpline(ISmoothingSpline[NdGridSplinePPForm, ty.Tuple[float, ...], NdGridDataType]):
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

    @classmethod
    def _prepare_data(cls, xdata, ydata, weights, smooth):
        xdata = ndgrid_prepare_data_sites(xdata, 'xdata')
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
            weights = ndgrid_prepare_data_sites(weights, 'weights')

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
        xi = ndgrid_prepare_data_sites(xi, 'xi')

        if len(xi) != self._ndim:
            raise ValueError(
                'xi ({}) and xdata ({}) dimensions mismatch'.format(len(xi), self._ndim))

        return self._spline.evaluate(xi)

    def _make_spline(self, smooth: ty.List[ty.Optional[float]]) -> ty.Tuple[NdGridSplinePPForm, ty.Tuple[float, ...]]:
        sizey = [1] + list(self._ydata.shape)
        ydata = self._ydata.reshape(sizey, order='F').copy()
        _smooth = []

        # Perform coordinatewise smoothing spline computing
        for i in range(self._ndim - 1, -1, -1):
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
