# -*- coding: utf-8 -*-

import pytest

import numpy as np
from scipy.interpolate import NdPPoly
import csaps


@pytest.mark.parametrize('x,y,w,p', [
    ([1, 2, 3], np.ones((10, 10)), None, None),
    ([[1], [1]], np.ones((1, 1)), None, None),
    ([[1, 2, 3], [1, 2, 3]], np.ones((4, 3)), None, None),
    ([[1, 2, 3], [1, 2, 3]], np.ones((3, 3, 3)), None, None),
    ([[1, 2, 3], [1, 2, 3]], np.ones((3, 3)), [1, 2, 3], None),
    ([[1, 2, 3], [1, 2, 3]], np.ones((3, 3)), [[1, 2, 3]], None),
    ([[1, 2, 3], [1, 2, 3]], np.ones((3, 3)), [[1, 2], [1, 2]], None),
    ([[1, 2, 3], [1, 2, 3]], np.ones((3, 3)), None, [0.5, 0.4, 0.2]),
    (np.array([[1, 2, 3], [4, 5, 6]]), np.ones((3, 3)), None, None),
    ([np.arange(6).reshape(2, 3), np.arange(6).reshape(2, 3)], np.ones((6, 6)), None, None),
])
def test_invalid_data(x, y, w, p):
    with pytest.raises((ValueError, TypeError)):
        csaps.NdGridCubicSmoothingSpline(x, y, w, p)


@pytest.mark.parametrize('x, xi', [
    (([1, 2, 3],), ([],)),
    (([1, 2, 3], [1, 2, 3]), ([1, 2], [])),
    (([1, 2, 3], [1, 2, 3], [1, 2, 3]), ([1, 2, 3], [1, 2, 3])),
    (([1, 2, 3], [1, 2, 3]), ([1, 2, 3], [1, 2, 3], [1, 2, 3])),
])
def test_invalid_evaluate_data(x, xi):
    np.random.seed(1234)

    y = np.random.randn(*tuple(len(xx) for xx in x))
    s = csaps.NdGridCubicSmoothingSpline(x, y)

    with pytest.raises((ValueError, TypeError)):
        s(xi)


@pytest.mark.parametrize('shape, coeffs_shape, order, pieces, ndim', [
    ((2,), (2, 1), (2,), (1,), 1),
    ((3,), (4, 2), (4,), (2,), 1),
    ((4,), (4, 3), (4,), (3,), 1),
    ((2, 4), (2, 4, 1, 3), (2, 4), (1, 3), 2),
    ((3, 4), (4, 4, 2, 3), (4, 4), (2, 3), 2),
    ((4, 4), (4, 4, 3, 3), (4, 4), (3, 3), 2),
    ((2, 3, 4), (2, 4, 4, 1, 2, 3), (2, 4, 4), (1, 2, 3), 3),
    ((3, 4, 5), (4, 4, 4, 2, 3, 4), (4, 4, 4), (2, 3, 4), 3),
    ((2, 3, 2, 6), (2, 4, 2, 4, 1, 2, 1, 5), (2, 4, 2, 4), (1, 2, 1, 5), 4),
    ((3, 4, 5, 6), (4, 4, 4, 4, 2, 3, 4, 5), (4, 4, 4, 4), (2, 3, 4, 5), 4),
])
def test_ndsplineppform(shape, coeffs_shape, order, pieces, ndim):
    x = tuple(np.arange(s) for s in shape)
    y = np.arange(float(np.prod(shape))).reshape(shape)

    ss = csaps.NdGridCubicSmoothingSpline(x, y).spline

    assert all(np.allclose(bi, xi) for bi, xi in zip(ss.breaks, x))
    assert ss.coeffs.shape == coeffs_shape
    assert ss.order == order
    assert ss.pieces == pieces
    assert ss.ndim == ndim
    assert ss.shape == shape


def test_surface(surface_data):
    xdata, ydata = surface_data

    sp = csaps.NdGridCubicSmoothingSpline(xdata, ydata)
    noisy_s = sp(xdata)

    assert isinstance(sp.smooth, tuple)
    assert len(sp.smooth) == 2
    assert isinstance(sp.spline, csaps.NdGridSplinePPForm)
    assert noisy_s.shape == ydata.shape


@pytest.mark.parametrize('shape, smooth', [
    ((4,), 0.0),
    ((4,), 0.5),
    ((4,), 1.0),

    ((4,), (0.0,)),
    ((4,), (0.5,)),
    ((4,), (1.0,)),

    ((4, 5), 0.0),
    ((4, 5), 0.5),
    ((4, 5), 1.0),
    ((4, 5), (0.0, 0.0)),
    ((4, 5), (0.0, 0.5)),
    ((4, 5), (0.5, 0.0)),
    ((4, 5), (0.5, 0.7)),
    ((4, 5), (1.0, 0.0)),
    ((4, 5), (0.0, 1.0)),
    ((4, 5), (1.0, 1.0)),

    ((4, 5, 6), 0.0),
    ((4, 5, 6), 0.5),
    ((4, 5, 6), 1.0),
    ((4, 5, 6), (0.0, 0.0, 0.0)),
    ((4, 5, 6), (0.5, 0.0, 0.0)),
    ((4, 5, 6), (0.5, 0.6, 0.0)),
    ((4, 5, 6), (0.0, 0.5, 0.6)),
    ((4, 5, 6), (0.4, 0.5, 0.6)),

    ((4, 5, 6, 7), 0.0),
    ((4, 5, 6, 7), 0.5),
    ((4, 5, 6, 7), 1.0),
    ((4, 5, 6, 7), (0.0, 0.0, 0.0, 0.0)),
    ((4, 5, 6, 7), (0.5, 0.0, 0.0, 0.0)),
    ((4, 5, 6, 7), (0.0, 0.5, 0.0, 0.0)),
    ((4, 5, 6, 7), (0.5, 0.6, 0.0, 0.0)),
    ((4, 5, 6, 7), (0.0, 0.5, 0.6, 0.0)),
    ((4, 5, 6, 7), (0.0, 0.5, 0.6, 0.7)),
    ((4, 5, 6, 7), (0.4, 0.5, 0.6, 0.7)),
])
def test_smooth_factor(shape, smooth):
    x = [np.arange(s) for s in shape]
    y = np.arange(0, np.prod(shape)).reshape(shape)

    sp = csaps.NdGridCubicSmoothingSpline(x, y, smooth=smooth)

    if isinstance(smooth, tuple):
        expected_smooth = smooth
    else:
        expected_smooth = tuple([smooth] * len(shape))

    assert sp.smooth == pytest.approx(expected_smooth)


@pytest.mark.parametrize('shape', [
    (2,),

    (2, 3),
    (2, 2),

    (2, 3, 4),
    (2, 2, 3),
    (2, 2, 2),

    (2, 3, 4, 5),
    (2, 2, 3, 4),
    (2, 2, 2, 3),
    (2, 2, 2, 2),

    (2, 3, 4, 5, 6),
    (2, 2, 3, 4, 5),
    (2, 2, 2, 3, 4),
    (2, 2, 2, 2, 3),
    (2, 2, 2, 2, 2),
])
def test_nd_2pt_array(shape: tuple):
    xdata = [np.arange(s) for s in shape]
    ydata = np.arange(0, np.prod(shape)).reshape(shape)

    sp = csaps.NdGridCubicSmoothingSpline(xdata, ydata, smooth=1.0)
    ydata_s = sp(xdata)

    assert ydata_s.shape == ydata.shape
    assert ydata_s == pytest.approx(ydata)


@pytest.mark.parametrize('shape', [
    (2,),
    (3,),
    (2, 3),
    (3, 4),
    (3, 2, 4),
    (3, 4, 5),
    (2, 4, 2, 6),
    (3, 4, 5, 6),
    (3, 2, 2, 6, 2),
    (3, 4, 5, 6, 7),
])
def test_nd_array(shape: tuple):
    xdata = [np.arange(s) for s in shape]
    ydata = np.arange(0, np.prod(shape)).reshape(shape)

    sp = csaps.NdGridCubicSmoothingSpline(xdata, ydata, smooth=1.0)
    ydata_s = sp(xdata)

    assert ydata_s == pytest.approx(ydata)


def test_auto_smooth_2d(ndgrid_2d_data):
    xy = ndgrid_2d_data.xy
    z = ndgrid_2d_data.z
    zi_expected = ndgrid_2d_data.zi
    smooth_expected = ndgrid_2d_data.smooth

    s = csaps.NdGridCubicSmoothingSpline(xy, z, smooth=None)
    zi = s(xy)

    assert s.smooth == pytest.approx(smooth_expected)
    assert zi == pytest.approx(zi_expected)


@pytest.mark.parametrize('nu', [
    None,
    (0, 0),
    (1, 1),
    (2, 2),
])
@pytest.mark.parametrize('extrapolate', [
    None,
    True,
    False,
])
def test_evaluate_nu_extrapolate(nu: tuple, extrapolate: bool):
    x = ([1, 2, 3, 4], [1, 2, 3, 4])
    xi = ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
    y = np.arange(4 * 4).reshape((4, 4))

    ss = csaps.NdGridCubicSmoothingSpline(x, y, smooth=1.0)
    y_ss = ss(xi, nu=nu, extrapolate=extrapolate)

    pp = NdPPoly(ss.spline.c, x)
    xx = tuple(np.meshgrid(*xi, indexing='ij'))
    y_pp = pp(xx, nu=nu, extrapolate=extrapolate)

    np.testing.assert_allclose(y_ss, y_pp, rtol=1e-05, atol=1e-08, equal_nan=True)
