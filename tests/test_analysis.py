# -*- coding: utf-8 -*-

import pytest
import csaps


def test_univariate_derivative(univariate_data):
    x = univariate_data.x
    y = univariate_data.y
    xi = univariate_data.xi

    yi_d1_expected = univariate_data.yi_d1
    yi_d2_expected = univariate_data.yi_d2

    spline = csaps.CubicSmoothingSpline(x, y, smooth=None).spline

    spline_d1: csaps.SplinePPForm = spline.derivative(nu=1)
    spline_d2: csaps.SplinePPForm = spline.derivative(nu=2)

    yi_d1 = spline_d1(xi)
    yi_d2 = spline_d2(xi)

    assert spline_d1.order == 3
    assert spline_d2.order == 2

    assert yi_d1 == pytest.approx(yi_d1_expected)
    assert yi_d2 == pytest.approx(yi_d2_expected)


def test_univariate_antiderivative(univariate_data):
    x = univariate_data.x
    y = univariate_data.y
    xi = univariate_data.xi

    yi_ad1_expected = univariate_data.yi_ad1

    spline = csaps.CubicSmoothingSpline(x, y, smooth=None).spline
    spline_ad1: csaps.SplinePPForm = spline.antiderivative(nu=1)

    yi_ad1 = spline_ad1(xi)

    assert spline_ad1.order == 5
    assert yi_ad1 == pytest.approx(yi_ad1_expected)


def test_univariate_integrate(univariate_data):
    x = univariate_data.x
    y = univariate_data.y
    integral_expected = univariate_data.integral

    spline = csaps.CubicSmoothingSpline(x, y, smooth=None).spline
    integral = spline.integrate(x[0], x[-1])

    assert integral == pytest.approx(integral_expected)
