"""
Type-hints and type vars
"""

from typing import Sequence, Tuple, TypeVar, Union
from collections import abc
from numbers import Number

from typing_extensions import TypeAlias
import numpy as np

UnivariateDataType: TypeAlias = Union[np.ndarray, Sequence[Number]]
MultivariateDataType: TypeAlias = Union[np.ndarray, abc.Sequence]
NdGridDataType: TypeAlias = Sequence[UnivariateDataType]

TData = TypeVar('TData', np.ndarray, Sequence[np.ndarray])
TProps = TypeVar('TProps', int, Tuple[int, ...])
TSmooth = TypeVar('TSmooth', float, Tuple[float, ...])
TXi = TypeVar('TXi', UnivariateDataType, NdGridDataType)
TNu = TypeVar('TNu', int, Tuple[int, ...])
TExtrapolate = TypeVar('TExtrapolate', bool, Union[bool, str])
TSpline = TypeVar('TSpline')
