"""
The base classes and interfaces
"""

from typing import Generic, Optional, Tuple
import abc

import numpy as np

from ._types import TData, TExtrapolate, TNu, TProps, TSmooth, TSpline, TXi, FloatNDArrayType


class ISplinePPForm(abc.ABC, Generic[TData, TProps]):
    """The interface class for spline representation in PP-form"""

    __module__ = 'csaps'

    @property
    @abc.abstractmethod
    def breaks(self) -> TData:
        """Returns the breaks for the spline

        Returns
        -------
        breaks : Union[np.ndarray, Tuple[np.ndarray, ...]]
            Breaks data
        """

    @property
    @abc.abstractmethod
    def coeffs(self) -> np.ndarray:
        """Returns the spline coefficients

        Returns
        -------
        coeffs : np.ndarray
            Coefficients n-d array
        """

    @property
    @abc.abstractmethod
    def order(self) -> TProps:
        """Returns the spline order

        Returns
        -------
        order : Union[int, Tuple[int, ...]]
            The spline order
        """

    @property
    @abc.abstractmethod
    def pieces(self) -> TProps:
        """Returns the spline pieces data

        Returns
        -------
        pieces : Union[int, Tuple[int, ...]]
            The spline pieces data
        """

    @property
    @abc.abstractmethod
    def ndim(self) -> int:
        """Returns the spline dimension count

        Returns
        -------
        ndim : int
            The spline dimension count
        """

    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Returns the source data shape

        Returns
        -------
        shape : tuple of int
            The source data shape
        """


class ISmoothingSpline(abc.ABC, Generic[TSpline, TSmooth, TXi, TNu, TExtrapolate]):
    """The interface class for smooting splines"""

    __module__ = 'csaps'

    @property
    @abc.abstractmethod
    def smooth(self) -> TSmooth:
        """Returns smoothing factor(s)"""

    @property
    @abc.abstractmethod
    def spline(self) -> TSpline:
        """Returns spline representation in PP-form"""

    @abc.abstractmethod
    def __call__(
        self,
        xi: TXi,
        nu: Optional[TNu] = None,
        extrapolate: Optional[TExtrapolate] = None,
    ) -> FloatNDArrayType:
        """Evaluates spline on the data sites"""
