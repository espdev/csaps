# -*- coding: utf-8 -*-

import typing as ty
import numpy as np


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

    if axis >= arr.ndim:
        raise ValueError(f'axis {axis} is out of array axes {arr.ndim}')

    tr_axes = list(range(arr.ndim))
    tr_axes.pop(axis)
    tr_axes.append(axis)

    new_shape = (np.prod(arr.shape) // arr.shape[axis], arr.shape[axis])

    return arr.transpose(tr_axes).reshape(new_shape)


def from_2d(arr: np.ndarray, shape: ty.Sequence[int], axis: int) -> np.ndarray:
    """Transforms 2-d NxM array to N-D array using given shape and axis

    Parameters
    ----------
    arr : np.ndarray
        2-D NxM array, where N is dimension and M is the nember of elements

    shape : tuple, list
        The shape of N-D array

    axis : int
        Axis that have been used for transform array shape

    Returns
    -------
    arrnd : np.ndarray
        N-D array

    Raises
    ------
    ValueError : axis is out of N-D array axes

    See Also
    --------
    to_2d

    Examples
    --------

    .. code-block:: python

        >>> shape = (2, 3, 4)
        >>> arr_2d = np.arange(1, np.prod(shape)+1).reshape(2*4, 3)
        >>> print(arr_2d)
        [[ 1,  2,  3],
         [ 4,  5,  6],
         [ 7,  8,  9],
         [10, 11, 12],
         [13, 14, 15],
         [16, 17, 18],
         [19, 20, 21],
         [22, 23, 24]]
        >>> arr = from_2d(arr_2d, shape=shape, axis=1)
        >>> print(arr)
        [[[ 1,  4,  7, 10],
          [ 2,  5,  8, 11],
          [ 3,  6,  9, 12]],

        [[13, 16, 19, 22],
         [14, 17, 20, 23],
         [15, 18, 21, 24]]]

    """

    arr = np.asarray(arr)
    ndim = len(shape)
    axis = ndim + axis if axis < 0 else axis

    if axis >= ndim:
        raise ValueError(f'axis {axis} is out of N-D array axes {ndim}')

    new_shape = list(shape)
    new_shape.pop(axis)
    new_shape.append(shape[axis])

    tr_axes = list(range(ndim))
    tr_axes.insert(axis, tr_axes.pop(-1))

    return arr.reshape(new_shape).transpose(tr_axes)
