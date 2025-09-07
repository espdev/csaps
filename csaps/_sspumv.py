"""
Univariate/multivariate cubic smoothing spline implementation
"""

from typing import Literal, cast
from functools import partial

import numpy as np
from scipy.interpolate import PPoly
import scipy.sparse as sp
import scipy.sparse.linalg as la

from ._base import ISmoothingSpline, ISplinePPForm
from ._reshape import prod, to_2d
from ._types import FloatNDArrayType, MultivariateDataType, UnivariateDataType

diags_csr = partial(sp.diags, format='csr')
vpad = partial(np.pad, pad_width=[(1, 1), (0, 0)], mode='constant')


class SplinePPForm(ISplinePPForm[np.ndarray, int], PPoly):
    """The base class for univariate/multivariate spline in piecewise polynomial form

    Piecewise polynomial in terms of coefficients and breakpoints.

    Notes
    -----

    Inherited from :py:class:`scipy.interpolate.PPoly`

    """

    __module__ = 'csaps'

    @property
    def breaks(self) -> np.ndarray:
        return self.x

    @property
    def coeffs(self) -> np.ndarray:
        return self.c

    @property
    def order(self) -> int:
        return self.c.shape[0]

    @property
    def pieces(self) -> int:
        return self.c.shape[1]

    @property
    def ndim(self) -> int:
        """Returns the number of spline dimensions

        The number of dimensions is product of shape without ``shape[self.axis]``.
        """
        shape = list(self.shape)
        shape.pop(self.axis)

        return prod(shape)

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the source data shape"""
        shape: list[int] = list(self.c.shape[2:])
        shape.insert(self.axis, self.c.shape[1] + 1)

        return tuple(shape)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f'{type(self).__name__}\n'
            f'  breaks: {self.breaks}\n'
            f'  coeffs shape: {self.coeffs.shape}\n'
            f'  data shape: {self.shape}\n'
            f'  axis: {self.axis}\n'
            f'  pieces: {self.pieces}\n'
            f'  order: {self.order}\n'
            f'  ndim: {self.ndim}\n'
        )


class CubicSmoothingSpline(
    ISmoothingSpline[
        SplinePPForm,
        float,
        UnivariateDataType,
        int,
        bool | Literal['periodic'],
    ]
):
    """Cubic smoothing spline

    The cubic spline implementation for univariate/multivariate data.

    Parameters
    ----------

    xdata : np.ndarray, sequence, vector-like
        X input 1-D data vector (data sites: ``x1 < x2 < ... < xN``)

    ydata : np.ndarray, vector-like, sequence[vector-like]
        Y input 1-D data vector or ND-array with shape[axis] equal of `xdata` size)

    weights : [*Optional*] np.ndarray, list
        Weights 1-D vector with size equal of ``xdata`` size

    smooth : [*Optional*] float
        Smoothing parameter in range [0, 1] where:
            - 0: The smoothing spline is the least-squares straight line fit
            - 1: The cubic spline interpolant with natural condition

    axis : [*Optional*] int
        Axis along which ``ydata`` is assumed to be varying.
        Meaning that for x[i] the corresponding values are np.take(ydata, i, axis=axis).
        By default, it is -1 (the last axis).

    normalizedsmooth : [*Optional*] bool
        If True, the smooth parameter is normalized such that results are invariant to xdata range
        and less sensitive to nonuniformity of weights and xdata clumping

        .. versionadded:: 1.1.0

    """

    __module__ = 'csaps'

    def __init__(
        self,
        xdata: UnivariateDataType,
        ydata: MultivariateDataType,
        weights: UnivariateDataType | None = None,
        smooth: float | None = None,
        axis: int = -1,
        normalizedsmooth: bool = False,
    ) -> None:
        x, y, w, shape, axis = self._prepare_data(xdata, ydata, weights, axis)
        coeffs, self._smooth = self._make_spline(x, y, w, smooth, shape, normalizedsmooth)
        self._spline = cast(SplinePPForm, SplinePPForm.construct_fast(coeffs, x, axis=axis))

    def __call__(
        self,
        x: UnivariateDataType,
        nu: int | None = None,
        extrapolate: bool | Literal['periodic'] | None = None,
    ) -> FloatNDArrayType:
        """Evaluate the spline for given data

        Parameters
        ----------

        x : 1-d array-like
            Points to evaluate the spline at.

        nu : [*Optional*] int
            Order of derivative to evaluate. Must be non-negative.

        extrapolate : [*Optional*] bool or 'periodic'
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs. If 'periodic',
            periodic extrapolation is used. Default is True.

        Notes
        -----

        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.

        """
        if nu is None:
            nu = 0
        return self._spline(x, nu=nu, extrapolate=extrapolate)

    @property
    def smooth(self) -> float:
        """Returns the smoothing factor

        Returns
        -------
        smooth : float
            Smoothing factor in the range [0, 1]
        """
        return self._smooth

    @property
    def spline(self) -> SplinePPForm:
        """Returns the spline description in `SplinePPForm` instance

        Returns
        -------
        spline : SplinePPForm
            The spline representation in :class:`SplinePPForm` instance
        """
        return self._spline

    @staticmethod
    def _prepare_data(xdata, ydata, weights, axis):
        xdata = np.asarray(xdata, dtype=np.float64)
        ydata = np.asarray(ydata, dtype=np.float64)

        if xdata.ndim > 1:
            raise ValueError("'xdata' must be a vector")
        if xdata.size < 2:
            raise ValueError("'xdata' must contain at least 2 data points.")

        axis = ydata.ndim + axis if axis < 0 else axis

        if ydata.shape[axis] != xdata.size:
            raise ValueError(
                f"'ydata' data must be a 1-D or N-D array with shape[{axis}] "
                f"that is equal to 'xdata' size ({xdata.size})"
            )

        # Rolling axis for using its shape while constructing coeffs array
        shape = np.rollaxis(ydata, axis).shape

        # Reshape ydata N-D array to 2-D NxM array where N is the data
        # dimension and M is the number of data points.
        ydata = to_2d(ydata, axis)

        if weights is None:
            weights = np.ones_like(xdata)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.size != xdata.size:
                raise ValueError('Weights vector size must be equal of xdata size')

        return xdata, ydata, weights, shape, axis

    @staticmethod
    def _compute_smooth(a, b):
        """
        The calculation of the smoothing spline requires the solution of a
        linear system whose coefficient matrix has the form p*A + (1-p)*B, with
        the matrices A and B depending on the data sites x. The default value
        of p makes p*trace(A) equal (1 - p)*trace(B).
        """

        def trace(m: sp.dia_matrix):
            return m.diagonal().sum()

        return 1.0 / (1.0 + trace(a) / (6.0 * trace(b)))

    @staticmethod
    def _normalize_smooth(x: np.ndarray, w: np.ndarray, smooth: float | None) -> float:
        """
        See the explanation here: https://github.com/espdev/csaps/pull/47
        """

        span = np.ptp(x)

        eff_x = 1 + (span**2) / np.sum(np.diff(x) ** 2)
        eff_w = np.sum(w) ** 2 / np.sum(w**2)
        k = 80 * (span**3) * (x.size**-2) * (eff_x**-0.5) * (eff_w**-0.5)

        s = 0.5 if smooth is None else smooth
        p = s / (s + (1 - s) * k)

        return p

    @staticmethod
    def _make_spline(x, y, w, smooth, shape, normalizedsmooth):
        pcount = x.size
        dx = np.diff(x)

        if not all(dx > 0):  # pragma: no cover
            raise ValueError("Items of 'xdata' vector must satisfy the condition: x1 < x2 < ... < xN")

        dy = np.diff(y, axis=1)
        dy_dx = dy / dx

        if pcount == 2:
            # The corner case for the data with 2 points (1 breaks interval)
            # In this case we have 2-ordered spline and linear interpolation in fact
            yi = y[:, 0][:, np.newaxis]

            c_shape = (2, pcount - 1) + shape[1:]
            c = np.vstack((dy_dx, yi)).reshape(c_shape)
            p = 1.0

            return c, p

        # Create diagonal sparse matrices
        diags_r = np.vstack((dx[1:], 2 * (dx[1:] + dx[:-1]), dx[:-1]))
        r = sp.spdiags(diags_r, [-1, 0, 1], pcount - 2, pcount - 2, format='csr')

        dx_recip = 1.0 / dx
        diags_qtw = np.vstack((dx_recip[:-1], -(dx_recip[1:] + dx_recip[:-1]), dx_recip[1:]))
        diags_sqrw_recip = 1.0 / np.sqrt(w)

        qtw = diags_csr(diags_qtw, [0, 1, 2], (pcount - 2, pcount)) @ diags_csr(diags_sqrw_recip, 0, (pcount, pcount))
        qtw = qtw @ qtw.T

        p = smooth

        if normalizedsmooth:
            p = CubicSmoothingSpline._normalize_smooth(x, w, smooth)
        elif smooth is None:
            p = CubicSmoothingSpline._compute_smooth(r, qtw)

        pp = 6.0 * (1.0 - p)

        # Solve linear system for the 2nd derivatives
        a = pp * qtw + p * r
        b = np.diff(dy_dx, axis=1).T

        u = la.spsolve(a, b)
        if u.ndim < 2:
            u = u[np.newaxis]
        if y.shape[0] == 1:
            u = u.T

        dx = dx[:, np.newaxis]

        d1 = np.diff(vpad(u), axis=0) / dx
        d2 = np.diff(vpad(d1), axis=0)

        diags_w_recip = 1.0 / w
        w = diags_csr(diags_w_recip, 0, (pcount, pcount))

        yi = y.T - (pp * w) @ d2
        pu = vpad(p * u)

        c1 = np.diff(pu, axis=0) / dx
        c2 = 3.0 * pu[:-1, :]
        c3 = np.diff(yi, axis=0) / dx - dx * (2.0 * pu[:-1, :] + pu[1:, :])
        c4 = yi[:-1, :]

        c_shape = (4, pcount - 1) + shape[1:]
        c = np.vstack((c1, c2, c3, c4)).reshape(c_shape)

        return c, p
