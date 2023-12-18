"""
Type-hints and type vars
"""

from typing import Sequence, Tuple, TypeVar, Union
from collections import abc
from numbers import Number

import numpy as np

UnivariateDataType = Union[np.ndarray, Sequence[Number]]
MultivariateDataType = Union[np.ndarray, abc.Sequence]
NdGridDataType = Sequence[UnivariateDataType]

TData = TypeVar('TData', np.ndarray, Sequence[np.ndarray])
TProps = TypeVar('TProps', int, Tuple[int, ...])
TSmooth = TypeVar('TSmooth', float, Tuple[float, ...])
TXi = TypeVar('TXi', UnivariateDataType, NdGridDataType)
TNu = TypeVar('TNu', int, Tuple[int, ...])
TExtrapolate = TypeVar('TExtrapolate', bool, Union[bool, str])
TSpline = TypeVar('TSpline')
