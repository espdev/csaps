# -*- coding: utf-8 -*-

import typing as t
import numpy as np


BreaksDataType = t.Union[
    # Univariate data sites
    np.ndarray,

    # Grid data sites
    t.Union[
        t.List[np.ndarray],
        t.Tuple[np.ndarray, ...]
    ]
]

UnivariateDataType = t.Union[
    np.ndarray,
    t.Sequence[t.Union[int, float]]
]

UnivariateVectorizedDataType = t.Union[
    UnivariateDataType,
    t.List['UnivariateVectorizedDataType']
]

MultivariateDataType = t.Union[
    np.ndarray,
    t.Sequence[UnivariateDataType]
]

NdGridDataType = t.Sequence[UnivariateDataType]
