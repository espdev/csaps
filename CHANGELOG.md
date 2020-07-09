# Changelog

## v1.0.0dev

* Use `PPoly` and `NdPPoly` base classes from SciPy interpolate module for `SplinePPForm` and `NdGridSplinePPForm` respectively.
* Remove deprecated classes `UnivariateCubicSmoothingSpline` and `MultivariateCubicSmoothingSpline`
* Update the documentation

**Notes**

In this release the spline representation (the array of spline coefficients) has been changed 
according to `PPoly`/`NdPPoly`. 
See SciPy [PPoly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PPoly.html) 
and [NdPPoly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NdPPoly.html) documentation for details.


## v0.11.0

* Internal re-design `SplinePPForm` and `NdGridSplinePPForm` classes [#17](https://github.com/espdev/csaps/issues/17):
    - Remove `shape` and `axis` properties and reshaping data in these classes
    - `NdGridSplinePPForm` coefficients array for 1D grid now is 1-d instead of 2-d
* Refactoring the code and decrease memory consumption
* Add `overload` type-hints for `csaps` function signatures

## v0.10.1

* Fix call of `numpy.pad` function for numpy <1.17 [#15](https://github.com/espdev/csaps/issues/15)

## v0.10.0

* Significant performance improvements for make/evaluate splines and memory consumption optimization
* Change format for storing spline coefficients (reshape coeffs array) to improve performance
* Add shape property to `SplinePPForm`/`NdGridSplinePPForm` and axis property to `SplinePPForm`
* Fix issues with the smoothing factor in nd-grid case: inverted ordering and unnable to use 0.0 value
* Update documentation

## v0.9.0

* Drop support of Python 3.5
* `weights`, `smooth` and `axis` arguments in `csaps` function are keyword-only now
* `UnivariateCubicSmoothingSpline` and `MultivariateCubicSmoothingSpline` classes are deprecated 
  and will be removed in 1.0.0 version. Use `CubicSmoothingSpline` instead.

## v0.8.0

* Add `csaps` function that can be used as the main API
* Refactor the internal structure of the package
* Add the [documentation](https://csaps.readthedocs.io)

**Attention**

This is the last version that supports Python 3.5. 
The next versions will support Python 3.6 or above.

## v0.7.0

* Add Generic-based type-hints and mypy-compatibility

## v0.6.1

* A slight refactoring and extra data copies removing

## v0.6.0

* Add "axis" parameter for univariate/multivariate cases

## v0.5.0

* Reorganize the project to package-based structure
* Add the interface class for all smoothing spline classes

## v0.4.2

* FIX: "smooth" value is 0.0 was not used

## v0.4.1

* First PyPI release
