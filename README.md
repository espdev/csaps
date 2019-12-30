# CSAPS: Cubic spline approximation (smoothing)

[![PyPI version](https://img.shields.io/pypi/v/csaps.svg)](https://pypi.python.org/pypi/csaps)
[![Build status](https://travis-ci.org/espdev/csaps.svg?branch=master)](https://travis-ci.org/espdev/csaps)
[![Coverage Status](https://coveralls.io/repos/github/espdev/csaps/badge.svg?branch=master)](https://coveralls.io/github/espdev/csaps?branch=master)
![Supported Python versions](https://img.shields.io/pypi/pyversions/csaps.svg)
[![License](https://img.shields.io/pypi/l/csaps.svg)](LICENSE)

This module provides cubic smoothing spline for univariate/multivariate/gridded data approximation.
The smoothing parameter can be calculated automatically or it can be set manually. 

The smoothing parameter should be in range `[0, 1]` where bounds are:
* 0: The smoothing spline is the least-squares straight line fit to the data
* 1: The natural cubic spline interpolant

The calculation of the smoothing spline requires the solution of a linear system 
whose coefficient matrix has the form `p*A + (1 - p)*B`, with the matrices `A` and `B` 
depending on the data sites `X`. The automatically computed smoothing parameter makes 
`p*trace(A) equal (1 - p)*trace(B)`.

## Installation

Python 3.5 or above is supported.

```
pip install csaps
```

The module depends only on NumPy and SciPy.

On Windows we highly recommend to use unofficial builds [NumPy+MKL](https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy) 
and [SciPy](https://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy) from Christoph Gohlke.

```
> mkdir depends

download numpy (https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy) to "depends" directory
download scipy (https://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy) to "depends" directory

> pip install --find-links depends/ csaps
```

## Smoothing univariate data

Univariate data are two vectors: X and Y with the same size. X is data sites, Y is data values.
For univariate case X-values must satisfy the condition: `x1 < x2 < ... < xN`.

You can use `UnivariateCubicSmoothingSpline` class for uivariate data smoothing.

```python
import numpy as np
import matplotlib.pyplot as plt
import csaps

np.random.seed(1234)

x = np.linspace(-5., 5., 25)
y = np.exp(-(x/2.5)**2) + (np.random.rand(25) - 0.2) * 0.3

sp = csaps.UnivariateCubicSmoothingSpline(x, y, smooth=0.85)

xs = np.linspace(x[0], x[-1], 150)
ys = sp(xs)

plt.plot(x, y, 'o', xs, ys, '-')
plt.show()
```

![csaps1d](https://user-images.githubusercontent.com/1299189/27611703-f3093c14-5b9b-11e7-9f18-6d0c3cc7633a.png)

### Smoothing weighted univariate data

The algorithm supports weighting. You can set weights vector that will determine 
weights for all data values:

```python
import csaps

x = [1., 2., 4., 6.]
y = [2., 4., 5., 7.]
w = [0.5, 1, 0.7, 1.2]

sp = csaps.UnivariateCubicSmoothingSpline(x, y, w)
...
```

### Smoothing univariate data with vectorization

The algorithm supports vectorization. You can compute smoothing splines for 
`X`, `Y` data where `X` is data site vector and `Y` is ND-array of data value vectors. 
The shape of `Y` array must be: `(d0, d1, ..., dN)` where `dN` must equal of `X` vector size.

In this case the smoothing splines will be computed for all `Y` data vectors at a time.

For example:

```python
import numpy as np
import csaps

# data sites
x = [1, 2, 3, 4]

# two data vectors
y = np.array([(2, 4, 6, 8), 
              (1, 3, 5, 7)])

sp = csaps.UnivariateCubicSmoothingSpline(x, y)

xi = np.linspace(1, 4, 10)
yi = sp(xi)

print(yi.shape)  # (2, 10)
assert yi.shape[:-1] == y.shape[:-1]
assert yi.shape[-1] == xi.size
```

**Important**:
The same weights vector and the same smoothing parameter will be used for all Y data.

## Smoothing multivariate data

We can easily smooth multivariate data using univariate smoothing spline, vectorization and parametrization.
We should define parametric data sites vector `t` with condition `t1 < t2 < ... < tN` and make spline for each dimension `X(t), Y(t), ..., M(t)`.

The module provides `MultivariateCubicSmoothingSpline` class for this case.

This class is a simple wrapper and it just automatically computes `t` vector by default and calls `UnivariateCubicSmoothingSpline`:

```python
# Construct multivariate spline from t and X, Y, Z, ..., M
sx = UnivariateCubicSmoothingSpline(t, data[x])
sy = UnivariateCubicSmoothingSpline(t, data[y])
sz = UnivariateCubicSmoothingSpline(t, data[z])
...
sm = UnivariateCubicSmoothingSpline(t, data[m])

# Or the same with using vectorization
sxyz_m = UnivariateCubicSmoothingSpline(t, data)
```
    
### Smoothing 3D noisy parametric curve example

In this example we make parametric 3D curve from `theta` parameter and compute spline from the same `theta` parameter.
In fact we could use `UnivariateCubicSmoothingSpline` with vectorization instead of `MultivariateCubicSmoothingSpline` in this case.

```python
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import csaps

n = 100
theta = np.linspace(-4 * np.pi, 4 * np.pi, n)
z = np.linspace(-2, 2, n)
r = z ** 2 + 1
np.random.seed(1234)
x = r * np.sin(theta) + np.random.randn(n) * 0.3
np.random.seed(5678)
y = r * np.cos(theta) + np.random.randn(n) * 0.3

data = np.vstack((x, y, z))

sp_theta = csaps.MultivariateCubicSmoothingSpline(data, theta, smooth=0.95)

# or the same
# sp_theta = csaps.UnivariateCubicSmoothingSpline(theta, data, smooth=0.95)

theta_i = np.linspace(-4 * np.pi, 4 * np.pi, 250)
data_i = sp_theta(theta_i)

xi = data_i[0, :]
yi = data_i[1, :]
zi = data_i[2, :]

fig = plt.figure()
ax = fig.gca(projection='3d')
plt.show()
```

<img width="663" alt="2019-04-04_02-28-52" src="https://user-images.githubusercontent.com/1299189/55519724-a75a6580-5681-11e9-8beb-2111664f8ee3.png">

## Smoothing ND-gridded data

The algorithm can make smoothing splines for ND-gridded data approximation.
In this case we use coordinatewise smoothing (tensor-product of univariate splines coefficients).

You can use `NdGridCubicSmoothingSpline` class for ND-gridded data smoothing. 
You also can set weights and smoothing parameters for each dimension.

Currently the implementation does not support vectorization for ND-gridded data.

### Smoothing 2D-gridded data (surface) example

```python
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import csaps

xdata = [np.linspace(-3, 3, 61), np.linspace(-3.5, 3.5, 51)]
i, j = np.meshgrid(*xdata, indexing='ij')

ydata = (3 * (1 - j)**2. * np.exp(-(j**2) - (i + 1)**2)
         - 10 * (j / 5 - j**3 - i**5) * np.exp(-j**2 - i**2)
         - 1 / 3 * np.exp(-(j + 1)**2 - i**2))

np.random.seed(12345)
noisy = ydata + (np.random.randn(*ydata.shape) * 0.75)

sp = csaps.NdGridCubicSmoothingSpline(xdata, noisy, smooth=0.988)
ysmth = sp(xdata)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(j, i, noisy, linewidths=0.5, color='r')
ax.scatter(j, i, noisy, s=5, c='r')

ax.plot_surface(j, i, ysmth, linewidth=0, alpha=1.0)

plt.show()
```

<img width="653" alt="2019-03-22_10-22-59" src="https://user-images.githubusercontent.com/1299189/54817564-2ff30200-4c8f-11e9-8afd-9055efcd6ea0.png">

## More examples

Please look through [csaps.ipynb](examples/csaps.ipynb) file for more examples.

## Testing

We use pytest and tox (on Travis CI) for testing.
Please see [test_csaps.py](tests/test_csaps.py) file.

## Algorithms and implementations

`csaps` is a Python modified port of MATLAB [CSAPS](https://www.mathworks.com/help/curvefit/csaps.html) function that is an implementation of 
Fortran routine SMOOTH from [PGS](http://pages.cs.wisc.edu/~deboor/pgs/) (originally written by Carl de Boor).

[csaps-cpp](https://github.com/espdev/csaps-cpp) C++11 Eigen based implementation of the algorithm.

## References

C. de Boor, A Practical Guide to Splines, Springer-Verlag, 1978.
