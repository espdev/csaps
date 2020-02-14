# -*- coding: utf-8 -*-

"""
The base classes and interfaces

"""

import abc
import typing as ty

import numpy as np

from ._types import TData, TProps, TSmooth, TXi, TSpline


class SplinePPFormBase(abc.ABC, ty.Generic[TData, TProps]):
    """The base class for spline representation in PP-form
    """

    @property
    @abc.abstractmethod
    def breaks(self) -> TData:
        """Returns breaks data

        Returns
        -------
        breaks : Union[np.ndarray, ty.Tuple[np.ndarray, ...]]
            Breaks data
        """

    @property
    @abc.abstractmethod
    def coeffs(self) -> np.ndarray:
        """Returns a spline coefficients 2-D array

        Returns
        -------
        coeffs : np.ndarray
            Coefficients 2-D array
        """

    @property
    @abc.abstractmethod
    def order(self) -> TProps:
        """Returns a spline order

        Returns
        -------
        order : ty.Union[int, ty.Tuple[int, ...]]
            Returns a spline order
        """

    @property
    @abc.abstractmethod
    def pieces(self) -> TProps:
        """Returns a spline pieces data

        Returns
        -------
        pieces : ty.Union[int, ty.Tuple[int, ...]]
            Returns a spline pieces data
        """

    @property
    @abc.abstractmethod
    def ndim(self) -> int:
        """Returns a spline dimension count

        Returns
        -------
        ndim : int
            A spline dimension count
        """

    @abc.abstractmethod
    def evaluate(self, xi: TData) -> np.ndarray:
        """Evaluates the spline for given data sites

        Parameters
        ----------
        xi : UnivariateDataType, NdGridDataType
            X data vector or list of vectors for multivariate spline

        Returns
        -------
        data : np.ndarray
            Interpolated/smoothed data
        """

    def __repr__(self):  # pragma: no cover
        return (
            f'{type(self).__name__}\n'
            f'  breaks: {self.breaks}\n'
            f'  coeffs: {self.coeffs.shape} shape\n'
            f'  pieces: {self.pieces}\n'
            f'  order: {self.order}\n'
            f'  ndim: {self.ndim}\n'
        )


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
