# -*- coding: utf-8 -*-

import numpy as np
import csaps


def test_auto_tdata():
    data = [
        (2, 4, 1, 3),  # X
        (1, 4, 3, 2),  # Y
        (3, 4, 1, 5),  # Z
    ]

    t = [0., 3.74165739, 8.10055633, 12.68313203]

    sp = csaps.MultivariateCubicSmoothingSpline(data)

    assert isinstance(sp.spline, csaps.SplinePPForm)
    assert 0 < sp.smooth < 1
    assert sp(t).shape == (3, 4)
    np.testing.assert_allclose(sp.t, t)
