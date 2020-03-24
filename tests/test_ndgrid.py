# -*- coding: utf-8 -*-

import pytest

import numpy as np
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


def test_surface():
    xdata = [np.linspace(-3, 3, 61), np.linspace(-3.5, 3.5, 51)]
    i, j = np.meshgrid(*xdata, indexing='ij')

    ydata = (3 * (1 - j)**2. * np.exp(-(j**2) - (i + 1)**2)
             - 10 * (j / 5 - j**3 - i**5) * np.exp(-j**2 - i**2)
             - 1 / 3 * np.exp(-(j + 1)**2 - i**2))

    np.random.seed(12345)
    noisy = ydata + (np.random.randn(*ydata.shape) * 0.75)

    sp = csaps.NdGridCubicSmoothingSpline(xdata, noisy)
    noisy_s = sp(xdata)

    assert isinstance(sp.smooth, tuple)
    assert len(sp.smooth) == 2
    assert isinstance(sp.spline, csaps.NdGridSplinePPForm)
    assert noisy_s.shape == noisy.shape


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
    (3,),
    (3, 4),
    (3, 4, 5),
    (3, 4, 5, 6),
    (3, 4, 5, 6, 7),
], ids=['1d', '2d', '3d', '4d', '5d'])
def test_nd_array(shape: tuple):
    xdata = [np.arange(s) for s in shape]
    ydata = np.arange(0, np.prod(shape)).reshape(shape)

    sp = csaps.NdGridCubicSmoothingSpline(xdata, ydata, smooth=1.0)
    ydata_s = sp(xdata)

    assert len(sp.spline.coeffs.shape) == len(ydata.shape)
    assert ydata_s == pytest.approx(ydata)
