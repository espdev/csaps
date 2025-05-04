"""
Type-hints and type vars
"""

from typing import Annotated, Literal, Sequence, TypeVar, Union

from typing_extensions import TypeAlias
import numpy as np
from numpy.typing import NDArray

FloatDType: TypeAlias = Union[np.float32, np.float64]
FloatNDArrayType: TypeAlias = NDArray[FloatDType]
Float1DArrayTupe: TypeAlias = Annotated[FloatNDArrayType, Literal['N']]

UnivariateDataType: TypeAlias = Union[Float1DArrayTupe, Sequence[float]]
MultivariateDataType: TypeAlias = Union[FloatNDArrayType, Sequence[Union[float, UnivariateDataType]]]
SequenceUnivariateDataType: TypeAlias = Sequence[UnivariateDataType]

TData = TypeVar('TData')
TProps = TypeVar('TProps')
TSmooth = TypeVar('TSmooth')
TXi = TypeVar('TXi')
TNu = TypeVar('TNu')
TExtrapolate = TypeVar('TExtrapolate')
TSpline = TypeVar('TSpline')
