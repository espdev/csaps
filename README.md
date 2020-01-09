# CSAPS: Cubic spline approximation (smoothing)

[![PyPI version](https://img.shields.io/pypi/v/csaps.svg)](https://pypi.python.org/pypi/csaps)
[![Documentation Status](https://readthedocs.org/projects/csaps/badge/?version=latest)](https://csaps.readthedocs.io/en/latest/?badge=latest)
[![Build status](https://travis-ci.org/espdev/csaps.svg?branch=master)](https://travis-ci.org/espdev/csaps)
[![Coverage Status](https://coveralls.io/repos/github/espdev/csaps/badge.svg?branch=master)](https://coveralls.io/github/espdev/csaps?branch=master)
![Supported Python versions](https://img.shields.io/pypi/pyversions/csaps.svg)
[![License](https://img.shields.io/pypi/l/csaps.svg)](LICENSE)

The module provides cubic smoothing spline for univariate/multivariate/nd-gridded data approximation.

## Installation

Python 3.5 or above is supported.

```
pip install -U csaps
```

The module depends only on NumPy and SciPy.

## Simple Examples

Here is a couple of examples of smoothing data.

An univariate data smoothing:

```python
import numpy as np
import matplotlib.pyplot as plt

from csaps import csaps

np.random.seed(1234)

x = np.linspace(-5., 5., 25)
y = np.exp(-(x/2.5)**2) + (np.random.rand(25) - 0.2) * 0.3
xs = np.linspace(x[0], x[-1], 150)

ys = csaps(x, y, xs, smooth=0.85)

plt.plot(x, y, 'o', xs, ys, '-')
plt.show()
```

![csaps1d](https://user-images.githubusercontent.com/1299189/27611703-f3093c14-5b9b-11e7-9f18-6d0c3cc7633a.png)

A surface data smoothing:

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

## Documentation

More examples of usage and the full documentation can be found at ReadTheDocs: https://csaps.readthedocs.io

## Testing

pytest, tox and Travis CI are used for testing. Please see [test_csaps.py](tests).

## Algorithms and implementations

**csaps** package is a Python modified port of MATLAB [CSAPS](https://www.mathworks.com/help/curvefit/csaps.html) function that is an implementation of 
Fortran routine SMOOTH from [PGS](http://pages.cs.wisc.edu/~deboor/pgs/) (originally written by Carl de Boor).

[csaps-cpp](https://github.com/espdev/csaps-cpp) C++11 Eigen based implementation of the algorithm.

## References

C. de Boor, A Practical Guide to Splines, Springer-Verlag, 1978.

## License

[MIT](https://choosealicense.com/licenses/mit/)
