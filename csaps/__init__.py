"""
Cubic spline approximation (smoothing)
"""

from csaps._base import ISmoothingSpline, ISplinePPForm
from csaps._shortcut import AutoSmoothingResult, csaps
from csaps._sspndg import NdGridCubicSmoothingSpline, NdGridSplinePPForm
from csaps._sspumv import CubicSmoothingSpline, SplinePPForm
from csaps._types import MultivariateDataType, SequenceUnivariateDataType, UnivariateDataType
from csaps._version import __version__

__all__ = [
    # Shortcut
    'csaps',
    'AutoSmoothingResult',
    # Classes
    'ISplinePPForm',
    'ISmoothingSpline',
    'SplinePPForm',
    'NdGridSplinePPForm',
    'CubicSmoothingSpline',
    'NdGridCubicSmoothingSpline',
    # Type-hints
    'UnivariateDataType',
    'MultivariateDataType',
    'SequenceUnivariateDataType',
    '__version__',
]
