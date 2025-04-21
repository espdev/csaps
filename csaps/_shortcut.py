"""
The module provided `csaps` shortcut function for smoothing data
"""

from typing import NamedTuple, Sequence, overload

import numpy as np

from ._base import ISmoothingSpline
from ._sspndg import NdGridCubicSmoothingSpline
from ._sspumv import CubicSmoothingSpline
from ._types import MultivariateDataType, SequenceUnivariateDataType, UnivariateDataType


class AutoSmoothingResult(NamedTuple):
    """The result for auto smoothing for `csaps` function"""

    values: MultivariateDataType
    """Smoothed data values"""

    smooth: float | Sequence[float | None]
    """The calculated smoothing parameter"""


# **************************************
# csaps signatures
#


@overload
def csaps(
    xdata: UnivariateDataType,
    ydata: MultivariateDataType,
    *,
    weights: UnivariateDataType | None = None,
    smooth: float | None = None,
    axis: int | None = None,
    normalizedsmooth: bool = False,
) -> ISmoothingSpline:  # pragma: no cover
    ...


@overload
def csaps(
    xdata: UnivariateDataType,
    ydata: MultivariateDataType,
    xidata: UnivariateDataType,
    *,
    weights: UnivariateDataType | None = None,
    axis: int | None = None,
    normalizedsmooth: bool = False,
) -> AutoSmoothingResult:  # pragma: no cover
    ...


@overload
def csaps(
    xdata: UnivariateDataType,
    ydata: MultivariateDataType,
    xidata: UnivariateDataType,
    *,
    smooth: float,
    weights: UnivariateDataType | None = None,
    axis: int | None = None,
    normalizedsmooth: bool = False,
) -> MultivariateDataType:  # pragma: no cover
    ...


@overload
def csaps(
    xdata: SequenceUnivariateDataType,
    ydata: MultivariateDataType,
    *,
    weights: SequenceUnivariateDataType | None = None,
    smooth: Sequence[float | None] | None = None,
    axis: int | None = None,
    normalizedsmooth: bool = False,
) -> ISmoothingSpline:  # pragma: no cover
    ...


@overload
def csaps(
    xdata: SequenceUnivariateDataType,
    ydata: MultivariateDataType,
    xidata: SequenceUnivariateDataType,
    *,
    weights: SequenceUnivariateDataType | None = None,
    axis: int | None = None,
    normalizedsmooth: bool = False,
) -> AutoSmoothingResult:  # pragma: no cover
    ...


@overload
def csaps(
    xdata: SequenceUnivariateDataType,
    ydata: MultivariateDataType,
    xidata: SequenceUnivariateDataType,
    *,
    smooth: Sequence[float | None],
    weights: SequenceUnivariateDataType | None = None,
    axis: int | None = None,
    normalizedsmooth: bool = False,
) -> MultivariateDataType:  # pragma: no cover
    ...


# **************************************
# csaps implementation


def csaps(
    xdata,
    ydata,
    xidata=None,
    *,
    weights=None,
    smooth=None,
    axis=None,
    normalizedsmooth=False,
):
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

    normalizedsmooth : [*Optional*] bool
        If True, the smooth parameter is normalized such that results are invariant to xdata range
        and less sensitive to nonuniformity of weights and xdata clumping

        .. versionadded:: 1.1.0

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
            - :class:`CubicSmoothingSpline` instance for univariate/multivariate data
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

    CubicSmoothingSpline
    NdGridCubicSmoothingSpline

    """

    umv = True
    if isinstance(xdata, Sequence):
        if len(xdata) and isinstance(xdata[0], (np.ndarray, Sequence)):
            umv = False

    if umv:
        axis = -1 if axis is None else axis
        sp = CubicSmoothingSpline(
            xdata=xdata,
            ydata=ydata,
            weights=weights,
            smooth=smooth,
            axis=axis,
            normalizedsmooth=normalizedsmooth,
        )
    else:
        sp = NdGridCubicSmoothingSpline(
            xdata=xdata,
            ydata=ydata,
            weights=weights,
            smooth=smooth,
            normalizedsmooth=normalizedsmooth,
        )

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
