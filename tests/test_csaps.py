# -*- coding: utf-8 -*-

import numpy as np
import csaps


def test_univariate_csaps():
    np.random.seed(1234)

    x = np.linspace(0, 2 * np.pi, 21)
    y = np.sin(x) + (np.random.rand(21) - .5) * .1

    sp = csaps.UnivariateCubicSmootingSpline(x, y)

    xi = np.linspace(x[0], x[-1], 121)
    yi = sp(xi)
