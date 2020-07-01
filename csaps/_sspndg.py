# -*- coding: utf-8 -*-

"""
ND-Gridded cubic smoothing spline implementation

"""

import collections.abc as c_abc
from numbers import Number
from typing import Tuple, Sequence, Optional, Union

import numpy as np
from scipy.interpolate import NdPPoly

from ._base import ISplinePPForm, ISmoothingSpline
from ._types import UnivariateDataType, NdGridDataType
from ._sspumv import SplinePPForm, CubicSmoothingSpline
from ._reshape import block_view


def ndgrid_prepare_data_vectors(data, name, min_size: int = 2) -> Tuple[np.ndarray, ...]:
    if not isinstance(data, c_abc.Sequence):
        raise TypeError(f"'{name}' must be a sequence of 1-d array-like (vectors) or scalars.")

    data = list(data)

    for axis, d in enumerate(data):
        d = np.asarray(d, dtype=np.float64)
        if d.ndim > 1:
            raise ValueError(f"All '{name}' elements must be a vector for axis {axis}.")
        if d.size < min_size:
            raise ValueError(f"'{name}' must contain at least {min_size} data points for axis {axis}.")
        data[axis] = d

    return tuple(data)


def _flatten_coeffs(spline: SplinePPForm):
    shape = list(spline.shape)
    shape.pop(spline.axis)
    c_shape = (spline.order * spline.pieces, int(np.prod(shape)))

    return spline.c.reshape(c_shape).T


class NdGridSplinePPForm(ISplinePPForm[Tuple[np.ndarray, ...], Tuple[int, ...]],
                         NdPPoly):
    """N-D grid spline representation in PP-form

    N-D grid spline is represented in piecewise tensor product polynomial form.
    """

    @property
    def breaks(self) -> Tuple[np.ndarray, ...]:
        return self.x

    @property
    def coeffs(self) -> np.ndarray:
        return self.c

    @property
    def order(self) -> Tuple[int, ...]:
        return self.c.shape[:self.c.ndim // 2]

    @property
    def pieces(self) -> Tuple[int, ...]:
        return self.c.shape[self.c.ndim // 2:]

    @property
    def ndim(self) -> int:
        return len(self.x)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(len(xi) for xi in self.x)

    def __call__(self,
                 x: Sequence[UnivariateDataType],
                 nu: Optional[Tuple[int, ...]] = None,
                 extrapolate: Optional[bool] = None) -> np.ndarray:
        """Evaluate the spline for given data

        Parameters
        ----------

        x : tuple of 1-d array-like
            The tuple of point values for each dimension to evaluate the spline at.

        nu : [*Optional*] tuple of int
            Orders of derivatives to evaluate. Each must be non-negative.

        extrapolate : [*Optional*] bool
            Whether to extrapolate to out-of-bounds points based on first and last
            intervals, or to return NaNs.

        Returns
        -------

        y : array-like
            Interpolated values. Shape is determined by replacing the
            interpolation axis in the original array with the shape of x.

        """
        x = ndgrid_prepare_data_vectors(x, 'x', min_size=1)

        if len(x) != self.ndim:
            raise ValueError(
                f"'x' sequence must have length {self.ndim} according to 'breaks'")

        x = tuple(np.meshgrid(*x, indexing='ij'))

        return super().__call__(x, nu, extrapolate)

    def __repr__(self):  # pragma: no cover
        return (
            f'{type(self).__name__}\n'
            f'  breaks: {self.breaks}\n'
            f'  coeffs shape: {self.coeffs.shape}\n'
            f'  data shape: {self.shape}\n'
            f'  pieces: {self.pieces}\n'
            f'  order: {self.order}\n'
            f'  ndim: {self.ndim}\n'
        )


class NdGridCubicSmoothingSpline(ISmoothingSpline[
                                     NdGridSplinePPForm,
                                     Tuple[float, ...],
                                     NdGridDataType,
                                     Tuple[int, ...],
                                     bool,
                                 ]):
    """N-D grid cubic smoothing spline

    Class implements N-D grid data smoothing (piecewise tensor product polynomial).

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
                 weights: Optional[Union[UnivariateDataType, NdGridDataType]] = None,
                 smooth: Optional[Union[float, Sequence[Optional[float]]]] = None) -> None:

        x, y, w, s = self._prepare_data(xdata, ydata, weights, smooth)
        coeffs, smooth = self._make_spline(x, y, w, s)

        self._spline = NdGridSplinePPForm.construct_fast(coeffs, x)
        self._smooth = smooth

    def __call__(self,
                 x: Union[NdGridDataType, Sequence[Number]],
                 nu: Optional[Tuple[int, ...]] = None,
                 extrapolate: Optional[bool] = None) -> np.ndarray:
        """Evaluate the spline for given data

        Parameters
        ----------

        x : tuple of 1-d array-like
            The tuple of point values for each dimension to evaluate the spline at.

        nu : [*Optional*] tuple of int
            Orders of derivatives to evaluate. Each must be non-negative.

        extrapolate : [*Optional*] bool
            Whether to extrapolate to out-of-bounds points based on first and last
            intervals, or to return NaNs.

        Returns
        -------

        y : array-like
            Interpolated values. Shape is determined by replacing the
            interpolation axis in the original array with the shape of x.

        """
        return self._spline(x, nu=nu, extrapolate=extrapolate)

    @property
    def smooth(self) -> Tuple[float, ...]:
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
        xdata = ndgrid_prepare_data_vectors(xdata, 'xdata')
        ydata = np.asarray(ydata)
        data_ndim = len(xdata)

        if ydata.ndim != data_ndim:
            raise ValueError(
                f"'ydata' must have dimension {data_ndim} according to 'xdata'")

        for axis, (yd, xs) in enumerate(zip(ydata.shape, map(len, xdata))):
            if yd != xs:
                raise ValueError(
                    f"'ydata' ({yd}) and xdata ({xs}) sizes mismatch for axis {axis}")

        if not weights:
            weights = [None] * data_ndim
        else:
            weights = ndgrid_prepare_data_vectors(weights, 'weights')

        if len(weights) != data_ndim:
            raise ValueError(
                f"'weights' ({len(weights)}) and 'xdata' ({data_ndim}) dimensions mismatch")

        for axis, (w, x) in enumerate(zip(weights, xdata)):
            if w is not None:
                if w.size != x.size:
                    raise ValueError(
                        f"'weights' ({w.size}) and 'xdata' ({x.size}) sizes mismatch for axis {axis}")

        if smooth is None:
            smooth = [None] * data_ndim

        if not isinstance(smooth, c_abc.Sequence):
            smooth = [float(smooth)] * data_ndim
        else:
            smooth = list(smooth)

        if len(smooth) != data_ndim:
            raise ValueError(
                'Number of smoothing parameter values must '
                f'be equal number of dimensions ({data_ndim})')

        return xdata, ydata, weights, smooth

    @staticmethod
    def _make_spline(xdata, ydata, weights, smooth):
        ndim = len(xdata)

        if ndim == 1:
            s = CubicSmoothingSpline(
                xdata[0], ydata, weights=weights[0], smooth=smooth[0])
            return s.spline.coeffs, (s.smooth,)

        shape = ydata.shape
        coeffs = ydata
        coeffs_shape = list(shape)

        smooths = []
        permute_axes = (ndim - 1, *range(ndim - 1))

        # computing coordinatewise smoothing spline
        for i in reversed(range(ndim)):
            if ndim > 2:
                coeffs = coeffs.reshape(np.prod(coeffs.shape[:-1]), coeffs.shape[-1])

            s = CubicSmoothingSpline(
                xdata[i], coeffs, weights=weights[i], smooth=smooth[i])

            smooths.append(s.smooth)
            coeffs = _flatten_coeffs(s.spline)

            if ndim > 2:
                coeffs_shape[-1] = s.spline.pieces * s.spline.order
                coeffs = coeffs.reshape(coeffs_shape)

            coeffs = coeffs.transpose(permute_axes)
            coeffs_shape = list(coeffs.shape)

        block = tuple(int(size - 1) for size in shape)
        coeffs = block_view(coeffs.squeeze(), block)

        return coeffs, tuple(reversed(smooths))
