# -*- coding: utf-8 -*-

"""
Cubic spline approximation (smoothing)

"""

from csaps._version import __version__  # noqa

from csaps._base import (
    SplinePPFormBase,
    ISmoothingSpline,
)
from csaps._sspumv import (
    SplinePPForm,
    CubicSmoothingSpline,
    UnivariateCubicSmoothingSpline,
    MultivariateCubicSmoothingSpline,
)
from csaps._sspndg import (
    NdGridSplinePPForm,
    NdGridCubicSmoothingSpline,
)
from csaps._types import (
    UnivariateDataType,
    MultivariateDataType,
    NdGridDataType,
)
from csaps._shortcut import csaps, AutoSmoothingResult

__all__ = [
    # Shortcut
    'csaps',
    'AutoSmoothingResult',

    # Classes
    'SplinePPFormBase',
    'ISmoothingSpline',
    'SplinePPForm',
    'NdGridSplinePPForm',
    'CubicSmoothingSpline',
    'UnivariateCubicSmoothingSpline',
    'MultivariateCubicSmoothingSpline',
    'NdGridCubicSmoothingSpline',

    # Type-hints
    'UnivariateDataType',
    'MultivariateDataType',
    'NdGridDataType',
]
