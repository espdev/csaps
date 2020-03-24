# -*- coding: utf-8 -*-

"""
ND-Gridded cubic smoothing spline implementation

"""

import collections.abc as c_abc
import typing as ty

import numpy as np

from ._base import SplinePPFormBase, ISmoothingSpline
from ._types import UnivariateDataType, NdGridDataType
from ._sspumv import SplinePPForm, CubicSmoothingSpline


def ndgrid_prepare_data_sites(data, name) -> ty.Tuple[np.ndarray, ...]:
    if not isinstance(data, c_abc.Sequence):
        raise TypeError(f"'{name}' must be a sequence of the vectors.")

    data = list(data)

    for i, di in enumerate(data):
        di = np.array(di, dtype=np.float64)
        if di.ndim > 1:
            raise ValueError(f"All '{name}' elements must be a vector.")
        if di.size < 2:
            raise ValueError(f"'{name}' must contain at least 2 data points.")
        data[i] = di

    return tuple(data)


class NdGridSplinePPForm(SplinePPFormBase[ty.Sequence[np.ndarray], ty.Tuple[int, ...]]):
    """N-D grid spline representation in PP-form

    Parameters
    ----------
    breaks : np.ndarray
        Breaks values 1-D array
    coeffs : np.ndarray
        Spline coefficients N-D array
    """

    def __init__(self, breaks: ty.Sequence[np.ndarray], coeffs: np.ndarray) -> None:
        self._breaks = breaks
        self._coeffs = coeffs
        self._pieces = tuple(x.size - 1 for x in breaks)
        self._ndim = len(breaks)

        if self._ndim > 1:
            self._order = tuple(s // p for s, p in zip(coeffs.shape, self._pieces))
        else:
            # the corner case for univariate spline that is represented as 1d-grid
            self._order = (coeffs.shape[1] // self._pieces[0], )

    @property
    def breaks(self) -> ty.Sequence[np.ndarray]:
        return self._breaks

    @property
    def coeffs(self) -> np.ndarray:
        return self._coeffs

    @property
    def order(self) -> ty.Tuple[int, ...]:
        return self._order

    @property
    def pieces(self) -> ty.Tuple[int, ...]:
        return self._pieces

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        """Returns the original data shape
        """
        return tuple(x.size for x in self.breaks)

    def evaluate(self, xi: ty.Sequence[np.ndarray]) -> np.ndarray:
        shape = tuple(x.size for x in xi)

        coeffs = self.coeffs
        coeffs_shape = list(coeffs.shape)

        d = self.ndim - 1
        permuted_axes = (d, *range(d))

        for i in reversed(range(self.ndim)):
            xii = xi[i]
            ndim = int(np.prod(coeffs_shape[:d]))

            if self.ndim > 2:
                coeffs = coeffs.reshape((ndim, self.pieces[i] * self.order[i]))

            coeffs = SplinePPForm(breaks=self.breaks[i], coeffs=coeffs).evaluate(xii)

            if self.ndim > 2:
                coeffs = coeffs.reshape((*coeffs_shape[:d], shape[i]))

            if self.ndim > 1:
                coeffs = coeffs.transpose(permuted_axes)
                coeffs_shape = list(coeffs.shape)

        return coeffs.reshape(shape)


class NdGridCubicSmoothingSpline(ISmoothingSpline[NdGridSplinePPForm, ty.Tuple[float, ...], NdGridDataType]):
    """ND-Gridded cubic smoothing spline

    Class implements ND-gridded data smoothing (piecewise tensor product polynomial).

    Parameters
    ----------

    xdata : list, tuple, Sequence[vector-like]
        X data site vectors for each dimensions. These vectors determine ND-grid.
        For example::

            # 2D grid
            x = [
                np.linspace(0, 5, 21),
                np.linspace(0, 6, 25),
            ]

    ydata : np.ndarray
        Y data ND-array with shape equal ``xdata`` vector sizes

    weights : [*Optional*] list, tuple, Sequence[vector-like]
        Weights data vector(s) for all dimensions or each dimension with
        size(s) equal to ``xdata`` sizes

    smooth : [*Optional*] float, Sequence[float]
        The smoothing parameter (or a sequence of parameters for each dimension) in range ``[0, 1]`` where:
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
        """Returns a tuple of smoothing parameters for each axis

        Returns
        -------
        smooth : Tuple[float, ...]
            The smoothing parameter in the range ``[0, 1]`` for each axis
        """
        return self._smooth

    @property
    def spline(self) -> NdGridSplinePPForm:
        """Returns the spline description in 'NdGridSplinePPForm' instance

        Returns
        -------
        spline : NdGridSplinePPForm
            The spline description in :class:`NdGridSplinePPForm` instance
        """
        return self._spline

    @classmethod
    def _prepare_data(cls, xdata, ydata, weights, smooth):
        xdata = ndgrid_prepare_data_sites(xdata, 'xdata')
        data_ndim = len(xdata)

        if ydata.ndim != data_ndim:
            raise ValueError(f'ydata must have dimension {data_ndim} according to xdata')

        for yd, xs in zip(ydata.shape, map(len, xdata)):
            if yd != xs:
                raise ValueError(f'ydata ({yd}) and xdata ({xs}) dimension size mismatch')

        if not weights:
            weights = [None] * data_ndim
        else:
            weights = ndgrid_prepare_data_sites(weights, 'weights')

        if len(weights) != data_ndim:
            raise ValueError(f'weights ({len(weights)}) and xdata ({data_ndim}) dimensions mismatch')

        for w, x in zip(weights, xdata):
            if w is not None:
                if w.size != x.size:
                    raise ValueError(f'weights ({w}) and xdata ({x}) dimension size mismatch')

        if smooth is None:
            smooth = [None] * data_ndim

        if not isinstance(smooth, c_abc.Sequence):
            smooth = [float(smooth)] * data_ndim
        else:
            smooth = list(smooth)

        if len(smooth) != data_ndim:
            raise ValueError(
                f'Number of smoothing parameter values must '
                f'be equal number of dimensions ({data_ndim})')

        return xdata, ydata, weights, smooth

    def __call__(self, xi: NdGridDataType) -> np.ndarray:
        """Evaluate the spline for given data
        """
        xi = ndgrid_prepare_data_sites(xi, 'xi')

        if len(xi) != self._ndim:  # pragma: no cover
            raise ValueError(f'xi ({len(xi)}) and xdata ({self._ndim}) dimensions mismatch')

        return self._spline.evaluate(xi)

    def _make_spline(self, smooth: ty.List[ty.Optional[float]]) -> ty.Tuple[NdGridSplinePPForm, ty.Tuple[float, ...]]:
        coeffs = self._ydata
        coeffs_shape = list(coeffs.shape)

        smooths = []
        permute_axes = (self._ndim - 1, *range(self._ndim - 1))

        # computing coordinatewise smoothing spline
        for i in reversed(range(self._ndim)):
            if self._ndim > 2:
                coeffs = coeffs.reshape(np.prod(coeffs.shape[:-1]), coeffs.shape[-1])

            s = CubicSmoothingSpline(
                self._xdata[i], coeffs, weights=self._weights[i], smooth=smooth[i])

            smooths.append(s.smooth)
            coeffs = s.spline.coeffs

            if self._ndim > 2:
                coeffs_shape[-1] = s.spline.pieces * s.spline.order
                coeffs = coeffs.reshape(coeffs_shape)

            if self._ndim > 1:
                coeffs = coeffs.transpose(permute_axes)
                coeffs_shape = list(coeffs.shape)

        return NdGridSplinePPForm(breaks=self._xdata, coeffs=coeffs), tuple(reversed(smooths))
