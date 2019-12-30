# -*- coding: utf-8 -*-

"""
Cubic spline approximation (smoothing)
"""

from csaps._version import __version__  # noqa

from csaps._base import (
    SplinePPForm,
    NdGridSplinePPForm,
    UnivariateCubicSmoothingSpline,
    MultivariateCubicSmoothingSpline,
    NdGridCubicSmoothingSpline,
)

from csaps._types import (
    UnivariateDataType,
    UnivariateVectorizedDataType,
    MultivariateDataType,
    NdGridDataType,
)

__all__ = [
    'SplinePPForm',
    'NdGridSplinePPForm',
    'UnivariateCubicSmoothingSpline',
    'MultivariateCubicSmoothingSpline',
    'NdGridCubicSmoothingSpline',

    # Type-hints
    'UnivariateDataType',
    'UnivariateVectorizedDataType',
    'MultivariateDataType',
    'NdGridDataType',
]
