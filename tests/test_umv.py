# -*- coding: utf-8 -*-

from itertools import chain, product, permutations

import numpy as np
from scipy.interpolate import CubicSpline
import pytest

import csaps


@pytest.mark.parametrize('x,y,w', [
    ([1], [2], None),
    ([1, 2, 3], [1, 2], None),
    ([(1, 2, 3), (1, 2, 3)], [1, 2, 3], None),
    ([1, 2, 3], [1, 2, 3], [1, 1]),
    ([1, 2, 3], [1, 2, 3], [1, 1, 1, 1]),
    ([1, 2, 3], [1, 2, 3], [(1, 1, 1), (1, 1, 1)]),
    ([1, 2, 3], [(1, 2, 3, 4), (1, 2, 3, 4)], None),
    ([1, 2, 3], np.ones((2, 4, 5)), None),
    ([1, 2, 3], np.ones((2, 4, 3)), np.ones((2, 4, 4))),
    ([1, 2, 3], [(1, 2, 3), (1, 2, 3)], [(1, 1, 1, 1), (1, 1, 1, 1)]),
    ([1, 2, 3], [(1, 2, 3), (1, 2, 3)], [(1, 1, 1), (1, 1, 1), (1, 1, 1)])
])
def test_invalid_data(x, y, w):
    with pytest.raises(ValueError):
        csaps.CubicSmoothingSpline(x, y, weights=w)


@pytest.mark.parametrize('y', [
    # 1D (2, )
    [2, 4],

    # 2D (2, 2)
    [(2, 4), (3, 5)],

    # 2D (3, 2)
    [(2, 4), (3, 5), (4, 6)],

    # 3D (2, 2, 2)
    [[(1, 2), (3, 4)],
     [(5, 6), (7, 8)]],

    # 1D (3, )
    [2, 4, 6],

    # 2D (2, 5)
    [(1, 2, 3, 4, 5), (3, 4, 5, 6, 7)],

    # 2D (3, 3)
    [(2, 4, 6), (3, 5, 7), (4, 6, 8)],

    # 3D (2, 2, 3)
    [[(2, 4, 6), (3, 5, 7)],
     [(2, 4, 6), (3, 5, 7)]],

    # 1D (4, )
    [2, 4, 6, 8],

    # 2D (2, 4)
    [(2, 4, 6, 8), (3, 5, 7, 9)],

    # 2D (3, 4)
    [(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)],

    # 3D (2, 2, 4)
    [[(2, 4, 6, 8), (3, 5, 7, 9)],
     [(2, 4, 6, 8), (3, 5, 7, 9)]],

    # 3D (2, 3, 4)
    [[(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)],
     [(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)]],

    # 3D (3, 2, 4)
    [[(2, 4, 6, 8), (3, 5, 7, 9)],
     [(2, 4, 6, 8), (3, 5, 7, 9)],
     [(2, 4, 6, 8), (3, 5, 7, 9)]],

    # 3D (3, 3, 4)
    [[(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)],
     [(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)],
     [(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)]],

    # 4D (2, 2, 2, 4)
    [[[(2, 4, 6, 8), (3, 5, 7, 9)], [(2, 4, 6, 8), (3, 5, 7, 9)]],
     [[(2, 4, 6, 8), (3, 5, 7, 9)], [(2, 4, 6, 8), (3, 5, 7, 9)]]],

    # 4D (3, 2, 2, 4)
    [[[(2, 4, 6, 8), (3, 5, 7, 9)], [(2, 4, 6, 8), (3, 5, 7, 9)]],
     [[(2, 4, 6, 8), (3, 5, 7, 9)], [(2, 4, 6, 8), (3, 5, 7, 9)]],
     [[(2, 4, 6, 8), (3, 5, 7, 9)], [(2, 4, 6, 8), (3, 5, 7, 9)]]],

    # 4D (3, 2, 3, 4)
    [[[(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)], [(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)]],
     [[(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)], [(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)]],
     [[(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)], [(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)]]],

    # 4D (3, 3, 3, 4)
    [[[(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)],
      [(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)],
      [(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)]],

     [[(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)],
      [(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)],
      [(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)]],

     [[(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)],
      [(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)],
      [(2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 10)]]],

])
def test_vectorize(y):
    x = np.arange(np.array(y).shape[-1])

    ys = csaps.CubicSmoothingSpline(x, y)(x)
    np.testing.assert_allclose(ys, y)


@pytest.mark.parametrize('shape, axis', chain(
    *map(product, [
        # shape
        [(2,), (4,)],
        permutations((2, 4), 2),
        permutations((3, 4, 5), 3),
        permutations((3, 4, 5, 6), 4)
    ], [
        # axis
        range(-1, 1),
        range(-2, 2),
        range(-3, 3),
        range(-4, 4),
    ])
))
def test_axis(shape, axis):
    y = np.arange(int(np.prod(shape))).reshape(shape)
    x = np.arange(np.array(y).shape[axis])

    s = csaps.CubicSmoothingSpline(x, y, axis=axis)

    ys = s(x)
    np.testing.assert_allclose(ys, y)

    ss = s.spline
    axis = len(shape) + axis if axis < 0 else axis
    ndim = int(np.prod(shape)) // shape[axis]
    order = 2 if len(x) < 3 else 4
    pieces = len(x) - 1
    coeffs_shape = (order, pieces) + shape[:axis] + shape[axis+1:]

    assert ss.breaks == pytest.approx(x)
    assert ss.coeffs.shape == coeffs_shape
    assert ss.axis == axis
    assert ss.order == order
    assert ss.pieces == pieces
    assert ss.ndim == ndim
    assert ss.shape == shape


def test_zero_smooth():
    x = [1., 2., 4., 6.]
    y = [2., 4., 5., 7.]

    sp = csaps.CubicSmoothingSpline(x, y, smooth=0.)

    assert sp.smooth == pytest.approx(0.)

    ys = sp(x)

    assert ys == pytest.approx([2.440677966101695,
                                3.355932203389830,
                                5.186440677966102,
                                7.016949152542373])


def test_auto_smooth(univariate_data):
    x, y, xi, yi_expected, *_, smooth_expected = univariate_data

    s = csaps.CubicSmoothingSpline(x, y, smooth=None)
    yi = s(xi)

    assert s.smooth == pytest.approx(smooth_expected)
    assert yi == pytest.approx(yi_expected)


@pytest.mark.parametrize('x,y,xi,yid', [
    ([1., 2.], [3., 4.], [1., 1.5, 2.], [3., 3.5, 4.]),
    ([1., 2., 3.], [3., 4., 5.], [1., 1.5, 2., 2.5, 3.], [3., 3.5, 4., 4.5, 5.]),
    ([1., 2., 4., 6.], [2., 4., 5., 7.], np.linspace(1., 6., 10), [
        2.2579392157892, 3.0231172855707, 3.6937304019483,
        4.21971044584031, 4.65026761247821, 5.04804510368134,
        5.47288175793241, 5.94265482897362, 6.44293945952166,
        6.95847986982311
    ]),
])
def test_npoints(x, y, xi, yid):
    sp = csaps.CubicSmoothingSpline(x, y)
    yi = sp(xi)

    np.testing.assert_allclose(yi, yid)


@pytest.mark.parametrize('w,yid', [
    ([0.5, 1, 0.7, 1.2], [
        2.39572102230177, 3.13781163365086, 3.78568993197139,
        4.28992448591238, 4.7009959256016, 5.08290363789967,
        5.49673867759808, 5.9600748344541, 6.45698622142886,
        6.97068522346297
    ])
])
def test_weighted(w, yid):
    x = [1., 2., 4., 6.]
    y = [2., 4., 5., 7.]
    xi = np.linspace(1., 6., 10)

    sp = csaps.CubicSmoothingSpline(x, y, weights=w)
    yi = sp(xi)

    np.testing.assert_allclose(yi, yid)


@pytest.mark.skip(reason='It may take a long time')
def test_big_vectorized():
    x = np.linspace(0, 10000, 10000)
    y = np.random.rand(1000, 10000)
    xi = np.linspace(0, 10000, 20000)

    csaps.CubicSmoothingSpline(x, y)(xi)


def test_cubic_bc_natural():
    np.random.seed(1234)
    x = np.linspace(0, 5, 20)
    xi = np.linspace(0, 5, 100)
    y = np.sin(x) + np.random.randn(x.size) * 0.3

    cs = CubicSpline(x, y, bc_type='natural')
    ss = csaps.CubicSmoothingSpline(x, y, smooth=1.0)

    y_cs = cs(xi)
    y_ss = ss(xi)

    assert cs.c == pytest.approx(ss.spline.c)
    assert y_cs == pytest.approx(y_ss)


@pytest.mark.parametrize('nu', [0, 1, 2])
@pytest.mark.parametrize('extrapolate', [None, True, False, 'periodic'])
def test_evaluate_nu_extrapolate(nu, extrapolate):
    x = [1, 2, 3, 4]
    xi = [0, 1, 2, 3, 4, 5]
    y = [1, 2, 3, 4]

    cs = CubicSpline(x, y)
    y_cs = cs(xi, nu=nu, extrapolate=extrapolate)

    ss = csaps.CubicSmoothingSpline(x, y, smooth=1.0)
    y_ss = ss(xi, nu=nu, extrapolate=extrapolate)

    np.testing.assert_allclose(y_ss, y_cs, rtol=1e-05, atol=1e-08, equal_nan=True)
