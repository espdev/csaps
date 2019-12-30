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
    ([[1, 2, 3], [1, 2, 3]], np.ones((3, 3)), None, [0.5, 0.4, 0.2])
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
    _ = sp(xdata)
