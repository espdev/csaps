# Changelog

## v1.3.3 (07.09.2025)

* Fix spsolve warning (set diag sparse matrix type to CSR to fix the warning)
* Add extrapolation section to tutorial documentation 
* Formatting code, type-hints (internal)


## v1.3.2 (15.04.2025)

* Remove `docs` extra dependencies from the package
* Refresh the documentation with Furo theme


## v1.3.1 (14.04.2025)

* Update readme and docs

## v1.3.0 (14.04.2025)

* Bump minimal Python version to 3.10
* Fix type annotations
* Fix checking types by mypy

## v1.2.1 (10.04.2025)

* Update dependencies
* Update the package classifiers

## v1.2.0 (30.06.2024)

* Bump minimal Python version to 3.9
* Use ruff as the code linter and formatter
* Update dependencies

## v1.1.0 (05.10.2021)

* Introduced optional `normalizedsmooth` argument to reduce dependence on xdata and weights [#47](https://github.com/espdev/csaps/pull/47)
* Update numpy and scipy dependency ranges

## v1.0.4 (04.05.2021)

* Bump numpy dependency version

## v1.0.3 (01.01.2021)

* Bump scipy dependency version
* Bump sphinx dependency version and use m2r2 sphinx extension instead of m2r
* Add Python 3.9 to classifiers list and to Travis CI
* Set development status classifier to "5 - Production/Stable"
* Happy New Year!

## v1.0.2 (19.07.2020)

* Fix using 'nu' argument when n-d grid spline evaluating [#32](https://github.com/espdev/csaps/pull/32)

## v1.0.1 (19.07.2020)

* Fix n-d grid spline evaluating performance regression [#31](https://github.com/espdev/csaps/pull/31)

## v1.0.0 (11.07.2020)

* Use `PPoly` and `NdPPoly` base classes from SciPy interpolate module for `SplinePPForm` and `NdGridSplinePPForm` respectively.
* Remove deprecated classes `UnivariateCubicSmoothingSpline` and `MultivariateCubicSmoothingSpline`
* Update the documentation

**Notes**

In this release the spline representation (the array of spline coefficients) has been changed 
according to `PPoly`/`NdPPoly`. 
See SciPy [PPoly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PPoly.html) 
and [NdPPoly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NdPPoly.html) documentation for details.


## v0.11.0 (28.03.2020)

* Internal re-design `SplinePPForm` and `NdGridSplinePPForm` classes [#17](https://github.com/espdev/csaps/issues/17):
    - Remove `shape` and `axis` properties and reshaping data in these classes
    - `NdGridSplinePPForm` coefficients array for 1D grid now is 1-d instead of 2-d
* Refactoring the code and decrease memory consumption
* Add `overload` type-hints for `csaps` function signatures

## v0.10.1 (19.03.2020)

* Fix call of `numpy.pad` function for numpy <1.17 [#15](https://github.com/espdev/csaps/issues/15)

## v0.10.0 (18.02.2020)

* Significant performance improvements for make/evaluate splines and memory consumption optimization
* Change format for storing spline coefficients (reshape coeffs array) to improve performance
* Add shape property to `SplinePPForm`/`NdGridSplinePPForm` and axis property to `SplinePPForm`
* Fix issues with the smoothing factor in nd-grid case: inverted ordering and unnable to use 0.0 value
* Update documentation

## v0.9.0 (21.01.2020)

* Drop support of Python 3.5
* `weights`, `smooth` and `axis` arguments in `csaps` function are keyword-only now
* `UnivariateCubicSmoothingSpline` and `MultivariateCubicSmoothingSpline` classes are deprecated 
  and will be removed in 1.0.0 version. Use `CubicSmoothingSpline` instead.

## v0.8.0 (13.01.2020)

* Add `csaps` function that can be used as the main API
* Refactor the internal structure of the package
* Add the [documentation](https://csaps.readthedocs.io)

**Attention**

This is the last version that supports Python 3.5. 
The next versions will support Python 3.6 or above.

## v0.7.0 (19.09.2019)

* Add Generic-based type-hints and mypy-compatibility

## v0.6.1 (13.09.2019)

* A slight refactoring and extra data copies removing

## v0.6.0 (12.09.2019)

* Add "axis" parameter for univariate/multivariate cases

## v0.5.0 (10.06.2019)

* Reorganize the project to package-based structure
* Add the interface class for all smoothing spline classes

## v0.4.2 (07.09.2019)

* FIX: "smooth" value is 0.0 was not used

## v0.4.1 (30.05.2019)

* First PyPI release
