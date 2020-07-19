# -*- coding: utf-8 -*-

import pytest
import numpy as np

from csaps._reshape import (  # noqa
    umv_coeffs_to_flatten,
    umv_coeffs_to_canonical,
    ndg_coeffs_to_flatten,
    ndg_coeffs_to_canonical,
)


@pytest.mark.parametrize('shape_canonical, shape_flatten, pieces', [
    ((2, 1), (1, 2), 1),
    ((3, 6), (1, 18), 6),
    ((4, 3), (1, 12), 3),
    ((4, 30), (1, 120), 30),
    ((4, 5, 2), (2, 20), 5),
    ((4, 6, 3), (3, 24), 6),
    ((4, 120, 53), (53, 480), 120),
])
def test_umv_coeffs_reshape(shape_canonical: tuple, shape_flatten: tuple, pieces: int):
    np.random.seed(1234)
    arr_canonical_expected = np.random.randint(0, 99, size=shape_canonical)

    arr_flatten = umv_coeffs_to_flatten(arr_canonical_expected)
    assert arr_flatten.shape == shape_flatten

    arr_canonical_actual = umv_coeffs_to_canonical(arr_flatten, pieces)
    np.testing.assert_array_equal(arr_canonical_actual, arr_canonical_expected)


@pytest.mark.parametrize('shape_canonical, shape_flatten, pieces', [
    # 1-d 2-ordered
    ((2, 3), (2, 3), (3,)),
    ((2, 4), (2, 4), (4,)),
    ((2, 5), (2, 5), (5,)),

    # 1-d 3-ordered
    ((3, 3), (3, 3), (3,)),
    ((3, 4), (3, 4), (4,)),
    ((3, 5), (3, 5), (5,)),

    # 1-d 4-ordered
    ((4, 3), (4, 3), (3,)),
    ((4, 4), (4, 4), (4,)),
    ((4, 5), (4, 5), (5,)),

    # 2-d {2,4}-ordered
    ((2, 4, 3, 4), (6, 16), (3, 4)),
    ((4, 2, 3, 3), (12, 6), (3, 3)),
    ((4, 2, 4, 3), (16, 6), (4, 3)),
    ((2, 4, 4, 4), (8, 16), (4, 4)),

    # 2-d {4,4}-ordered
    ((4, 4, 3, 3), (12, 12), (3, 3)),

    # 3-d {4,4,4}-ordered
    ((4, 4, 4, 3, 3, 3), (12, 12, 12), (3, 3, 3)),
    ((4, 4, 4, 3, 5, 7), (12, 20, 28), (3, 5, 7)),
])
def test_ndg_coeffs_reshape(shape_canonical: tuple, shape_flatten: tuple, pieces: tuple):
    np.random.seed(1234)
    arr_canonical_expected = np.random.randint(0, 99, size=shape_canonical)

    arr_flatten = ndg_coeffs_to_flatten(arr_canonical_expected)
    assert arr_flatten.shape == shape_flatten

    arr_canonical_actual = ndg_coeffs_to_canonical(arr_flatten, pieces)
    np.testing.assert_array_equal(arr_canonical_actual, arr_canonical_expected)
