# -*- coding: utf-8 -*-

"""
Type-hints and type vars

"""

from collections import abc
from typing import Union, Sequence, Tuple, TypeVar
from numbers import Number
import numpy as np


UnivariateDataType = Union[np.ndarray, Sequence[Number]]
MultivariateDataType = Union[np.ndarray, abc.Sequence]
NdGridDataType = Sequence[UnivariateDataType]

TData = TypeVar('TData', np.ndarray, Sequence[np.ndarray])
TProps = TypeVar('TProps', int, Tuple[int, ...])
TSmooth = TypeVar('TSmooth', float, Tuple[float, ...])
TXi = TypeVar('TXi', UnivariateDataType, NdGridDataType)
TSpline = TypeVar('TSpline')
