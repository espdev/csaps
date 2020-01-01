# -*- coding: utf-8 -*-

import pytest
import numpy as np

from csaps import csaps, SmoothingResult, UnivariateCubicSmoothingSpline, NdGridCubicSmoothingSpline


@pytest.fixture(scope='module')
def curve():
    np.random.seed(12345)

    x = np.linspace(-5., 5., 25)
    y = np.exp(-(x / 2.5) ** 2) + (np.random.rand(25) - 0.2) * 0.3
    return x, y


@pytest.fixture(scope='module')
def surface():
    np.random.seed(12345)

    x = [np.linspace(-3, 3, 61), np.linspace(-3.5, 3.5, 51)]
    i, j = np.meshgrid(*x, indexing='ij')

    y = (3 * (1 - j) ** 2. * np.exp(-(j ** 2) - (i + 1) ** 2)
         - 10 * (j / 5 - j ** 3 - i ** 5) * np.exp(-j ** 2 - i ** 2)
         - 1 / 3 * np.exp(-(j + 1) ** 2 - i ** 2))
    y += np.random.randn(*y.shape) * 0.75
    return x, y


@pytest.fixture
def data(curve, surface, request):
    if request.param == 'univariate':
        x, y = curve
        xi = np.linspace(x[0], x[-1], 150)
        return x, y, xi, 0.85, UnivariateCubicSmoothingSpline

    elif request.param == 'ndgrid':
        x, y = surface

        return x, y, x, [0.85, 0.85], NdGridCubicSmoothingSpline


@pytest.mark.parametrize('data', [
    'univariate',
    'ndgrid',
], indirect=True)
def test_shortcut_output(data):
    x, y, xi, smooth, sp_cls = data

    yi = csaps(x, y, xi, smooth=smooth)
    assert isinstance(yi, np.ndarray)

    smoothed_data = csaps(x, y, xi)
    assert isinstance(smoothed_data, SmoothingResult)

    sp = csaps(x, y)
    assert isinstance(sp, sp_cls)


@pytest.mark.parametrize('smooth, cls', [
    (0.85, np.ndarray),
    ([0.85, 0.85], np.ndarray),
    (None, SmoothingResult),
    ([None, 0.85], SmoothingResult),
    ([0.85, None], SmoothingResult),
    ([None, None], SmoothingResult),
])
def test_shortcut_ndgrid_smooth_output(surface, smooth, cls):
    x, y = surface

    output = csaps(x, y, x, smooth=smooth)
    assert isinstance(output, cls)
