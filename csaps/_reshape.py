import functools
from itertools import chain
import operator

import numpy as np
from numpy.lib.stride_tricks import as_strided


def prod(x):
    """Product of a list/tuple of numbers; ~40x faster vs np.prod for Python tuples"""
    if len(x) == 0:
        return 1
    return functools.reduce(operator.mul, x)


def to_2d(arr: np.ndarray, axis: int) -> np.ndarray:
    """Transforms the shape of N-D array to 2-D NxM array

    The function transforms N-D array to 2-D NxM array along given axis,
    where N is dimension and M is the nember of elements.

    The function does not create a copy.

    Parameters
    ----------
    arr : np.array
        N-D array

    axis : int
        Axis that will be used for transform array shape

    Returns
    -------
    arr2d : np.ndarray
        2-D NxM array view

    Raises
    ------
    ValueError : axis is out of array axes

    See Also
    --------
    from_2d

    Examples
    --------

    .. code-block:: python

        >>> shape = (2, 3, 4)
        >>> arr = np.arange(1, np.prod(shape)+1).reshape(shape)
        >>> arr_2d = to_2d(arr, axis=1)
        >>> print(arr)
        [[[ 1  2  3  4]
          [ 5  6  7  8]
          [ 9 10 11 12]]

         [[13 14 15 16]
          [17 18 19 20]
          [21 22 23 24]]]
        >>> print(arr_2d)
        [[ 1  5  9]
         [ 2  6 10]
         [ 3  7 11]
         [ 4  8 12]
         [13 17 21]
         [14 18 22]
         [15 19 23]
         [16 20 24]]

    """

    arr = np.asarray(arr)
    axis = arr.ndim + axis if axis < 0 else axis

    if axis >= arr.ndim:  # pragma: no cover
        raise ValueError(f'axis {axis} is out of array axes {arr.ndim}')

    tr_axes = list(range(arr.ndim))
    tr_axes.pop(axis)
    tr_axes.append(axis)

    new_shape = (np.prod(arr.shape) // arr.shape[axis], arr.shape[axis])

    return arr.transpose(tr_axes).reshape(new_shape)


def umv_coeffs_to_canonical(arr: np.ndarray, pieces: int):
    """

    Parameters
    ----------
    arr : array
        The 2-d array with shape (n, m) where:

            n -- the number of spline dimensions (1 for univariate)
            m -- order * pieces

    pieces : int
        The number of pieces

    Returns
    -------
    arr_view : array view
        The 2-d or 3-d array view with shape (k, p) or (k, p, n) where:

            k -- spline order
            p -- the number of spline pieces
            n -- the number of spline dimensions (multivariate case)

    """

    ndim: int = arr.shape[0]
    order: int = arr.shape[1] // pieces

    shape: tuple[int, ...]
    strides: tuple[int, ...]

    if ndim == 1:
        shape = (order, pieces)
        strides = (arr.strides[1] * pieces, arr.strides[1])
    else:
        shape = (order, pieces, ndim)
        strides = (arr.strides[1] * pieces, arr.strides[1], arr.strides[0])

    return as_strided(arr, shape=shape, strides=strides)


def umv_coeffs_to_flatten(arr: np.ndarray):
    """

    Parameters
    ----------
    arr : array
        The 2-d or 3-d array with shape (k, m) or (k, m, n) where:

            k -- the spline order
            m -- the number of spline pieces
            n -- the number of spline dimensions (multivariate case)

    Returns
    -------
    arr_view : array view
        The array 2-d view with shape (1, k * m) or (n, k * m)

    """

    if arr.ndim == 2:
        arr_view = arr.ravel()[np.newaxis]
    elif arr.ndim == 3:
        shape = (arr.shape[2], prod(arr.shape[:2]))
        strides = arr.strides[:-3:-1]
        arr_view = as_strided(arr, shape=shape, strides=strides)
    else:  # pragma: no cover
        raise ValueError(f'The array ndim must be 2 or 3, but given array has ndim={arr.ndim}.')

    return arr_view


def ndg_coeffs_to_canonical(arr: np.ndarray, pieces: tuple[int, ...]) -> np.ndarray:
    """Returns array canonical view for given n-d grid coeffs flatten array

    Creates n-d array canonical view with shape (k0, ..., kn, p0, ..., pn) for given
    array with shape (m0, ..., mn) and pieces (p0, ..., pn).

    Parameters
    ----------
    arr : array
        The input array with shape (m0, ..., mn)
    pieces : tuple
        The number of pieces (p0, ..., pn)

    Returns
    -------
    arr_view : array view
        The canonical view for given array with shape (k0, ..., kn, p0, ..., pn)

    """

    if arr.ndim > len(pieces):
        return arr

    shape = tuple(sz // p for sz, p in zip(arr.shape, pieces)) + pieces
    strides = tuple(st * p for st, p in zip(arr.strides, pieces)) + arr.strides

    return as_strided(arr, shape=shape, strides=strides)


def ndg_coeffs_to_flatten(arr: np.ndarray):
    """Creates flatten array view for n-d grid coeffs canonical array

    For example for input array (4, 4, 20, 30) will be created the flatten view (80, 120)

    Parameters
    ----------
    arr : array
        The input array with shape (k0, ..., kn, p0, ..., pn) where:

            ``k0, ..., kn`` -- spline orders
            ``p0, ..., pn`` -- spline pieces

    Returns
    -------
    arr_view : array view
        Flatten view of array with shape (m0, ..., mn)

    """

    if arr.ndim == 2:
        return arr

    ndim = arr.ndim // 2
    axes = tuple(chain.from_iterable(zip(range(ndim), range(ndim, arr.ndim))))
    shape = tuple(prod(arr.shape[i::ndim]) for i in range(ndim))

    return arr.transpose(axes).reshape(shape)
