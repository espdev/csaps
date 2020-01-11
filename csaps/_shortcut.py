# -*- coding: utf-8 -*-

"""
The module provised `csaps` shortcut function for smoothing data

"""

from collections import abc as c_abc
from typing import Optional, Union, Sequence, NamedTuple

import numpy as np

from csaps._base import ISmoothingSpline
from csaps._sspumv import UnivariateCubicSmoothingSpline
from csaps._sspndg import ndgrid_prepare_data_sites, NdGridCubicSmoothingSpline
from csaps._types import (
    UnivariateDataType,
    UnivariateVectorizedDataType,
    NdGridDataType,
)

_XDataType = Union[UnivariateDataType, NdGridDataType]
_YDataType = Union[UnivariateVectorizedDataType, np.ndarray]
_XiDataType = Optional[Union[UnivariateDataType, NdGridDataType]]
_WeightsDataType = Optional[Union[UnivariateDataType, NdGridDataType]]
_SmoothDataType = Optional[Union[float, Sequence[Optional[float]]]]

AutoSmoothingResult = NamedTuple('AutoSmoothingResult', [
    ('values', _YDataType),
    ('smooth', _SmoothDataType),
])
"""The result for auto smoothing for `csaps` function"""

_ReturnType = Union[
    _YDataType,
    AutoSmoothingResult,
    ISmoothingSpline,
]


def csaps(xdata: _XDataType,
          ydata: _YDataType,
          xidata: _XiDataType = None,
          weights: _WeightsDataType = None,
          smooth: _SmoothDataType = None,
          axis: Optional[int] = None) -> _ReturnType:
    """Smooths the univariate/multivariate/gridded data or computes the corresponding splines

    This function might be used as the main API for smoothing any data.

    Parameters
    ----------

    xdata : np.ndarray, array-like
        The data sites ``x1 < x2 < ... < xN``:
            - 1-D data vector/sequence (array-like) for univariate/multivariate ``ydata`` case
            - The sequence of 1-D data vectors for nd-gridded ``ydata`` case

    ydata : np.ndarray, array-like
        The data values:
            - 1-D data vector/sequence (array-like) for univariate data case
            - N-D array/array-like for multivariate data case
            - N-D array for nd-gridded data case

    xidata : [*Optional*] np.ndarray, array-like, Sequence[array-like]
        The data sites for output smoothed data:
            - 1-D data vector/sequence (array-like) for univariate/multivariate ``ydata`` case
            - The sequence of 1-D data vectors for nd-gridded ``ydata`` case

        If this argument was not set, the function will return computed spline
        for given data in :class:`ISmoothingSpline` object.

    weights : [*Optional*] np.ndarray, array-like, Sequence[array-like]
        The weights data vectors:
            - 1-D data vector/sequence (array-like) for univariate/multivariate ``ydata`` case
            - The sequence of 1-D data vectors for nd-gridded ``ydata`` case

    smooth : [*Optional*] float, Sequence[float]
        The smoothing factor value(s):
            - float value in the range ``[0, 1]`` for univariate/multivariate ``ydata`` case
            - the sequence of float in the range ``[0, 1]`` or None for nd-gridded ``ydata`` case

        If this argument was not set or None or sequence with None-items, the function will return
        named tuple :class:`AutoSmoothingResult` with computed smoothed data values and smoothing factor value(s).

    axis : [*Optional*] int
        The ``ydata`` axis. Axis along which ``ydata`` is assumed to be varying.
        If this argument was not set the last axis will be used (``axis == -1``).

        .. note::
            Currently, `axis` will be ignored for nd-gridded ``ydata`` case.

    Returns
    -------

    yidata : np.ndarray
        Smoothed data values if ``xidata`` and ``smooth`` were set.

    autosmoothing_result : AutoSmoothingResult
        The named tuple object with two fileds:
            - 'values' -- smoothed data values
            - 'smooth' -- computed smoothing factor

        This result will be returned if ``xidata`` was set and ``smooth`` was not set.

    ssp_obj : ISmoothingSpline
        Smoothing spline object if ``xidata`` was not set:
            - :class:`UnivariateCubicSmoothingSpline` instance for univariate/multivariate data
            - :class:`NdGridCubicSmoothingSpline` instance for nd-gridded data

    Examples
    --------

    Univariate data smoothing

    .. code-block:: python

        import numpy as np
        from csaps import csaps

        x = np.linspace(-5., 5., 25)
        y = np.exp(-(x/2.5)**2) + (np.random.rand(25) - 0.2) * 0.3
        xi = np.linspace(-5., 5., 150)

        # Smooth data with smoothing factor 0.85
        yi = csaps(x, y, xi, smooth=0.85)

        # Smooth data and compute smoothing factor automatically
        yi, smooth = csaps(x, y, xi)

        # Do not evaluate the spline, only compute it
        sp = csaps(x, y, smooth=0.98)

    See Also
    --------

    UnivariateCubicSmoothingSpline
    NdGridCubicSmoothingSpline

    """

    if isinstance(xdata, c_abc.Sequence):
        try:
            ndgrid_prepare_data_sites(xdata, 'xdata')
        except ValueError:
            umv = True
        else:
            umv = False
    else:
        umv = True

    if umv:
        axis = -1 if axis is None else axis
        sp = UnivariateCubicSmoothingSpline(xdata, ydata, weights, smooth, axis)
    else:
        sp = NdGridCubicSmoothingSpline(xdata, ydata, weights, smooth)

    if xidata is None:
        return sp

    yidata = sp(xidata)

    auto_smooth = smooth is None
    if isinstance(smooth, Sequence):
        auto_smooth = any(sm is None for sm in smooth)

    if auto_smooth:
        return AutoSmoothingResult(yidata, sp.smooth)
    else:
        return yidata
