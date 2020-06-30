# -*- coding: utf-8 -*-

"""
The base classes and interfaces

"""

import abc
import typing as ty

import numpy as np

from ._types import TData, TProps, TSmooth, TXi, TSpline


class ISplinePPForm(abc.ABC, ty.Generic[TData, TProps]):
    """The interface class for spline representation in PP-form
    """

    @property
    @abc.abstractmethod
    def breaks(self) -> TData:
        """Returns the breaks for the spline

        Returns
        -------
        breaks : Union[np.ndarray, ty.Tuple[np.ndarray, ...]]
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
        order : ty.Union[int, ty.Tuple[int, ...]]
            The spline order
        """

    @property
    @abc.abstractmethod
    def pieces(self) -> TProps:
        """Returns the spline pieces data

        Returns
        -------
        pieces : ty.Union[int, ty.Tuple[int, ...]]
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
    def shape(self) -> ty.Tuple[int]:
        """Returns the data shape

        Returns
        -------
        ndim : int
            The data shape
        """


class ISmoothingSpline(abc.ABC, ty.Generic[TSpline, TSmooth, TXi]):
    """The interface class for smooting splines
    """

    @property
    @abc.abstractmethod
    def smooth(self) -> TSmooth:
        """Returns smoothing parameter(s)
        """

    @property
    @abc.abstractmethod
    def spline(self) -> TSpline:
        """Returns spline representation in PP-form
        """

    @abc.abstractmethod
    def __call__(self, xi: TXi) -> np.ndarray:
        """Evaluates spline on the data sites
        """
