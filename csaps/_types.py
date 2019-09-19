# -*- coding: utf-8 -*-

import typing as ty
import numbers
import numpy as np


UnivariateDataType = ty.Union[
    np.ndarray,
    ty.Sequence[numbers.Number]
]

UnivariateVectorizedDataType = ty.Union[
    UnivariateDataType,
    # FIXME: mypy does not support recursive types
    # https://github.com/python/mypy/issues/731
    # ty.Sequence['UnivariateVectorizedDataType']
]

MultivariateDataType = ty.Union[
    np.ndarray,
    ty.Sequence[UnivariateDataType]
]

NdGridDataType = ty.Sequence[UnivariateDataType]
