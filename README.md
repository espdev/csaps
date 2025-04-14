<p align="center">
  <a href="https://github.com/espdev/csaps"><img src="https://user-images.githubusercontent.com/1299189/76571441-8d97e400-64c8-11ea-8c05-58850f8311a1.png" alt="csaps" width="400" /></a><br>
</p>

<p align="center">
  <a href="https://pypi.python.org/pypi/csaps"><img src="https://img.shields.io/pypi/v/csaps.svg" alt="PyPI version" /></a>
  <a href="https://pypi.python.org/pypi/csaps"><img src="https://img.shields.io/pypi/pyversions/csaps.svg" alt="Supported Python versions" /></a>
  <a href="https://github.com/espdev/csaps"><img src="https://github.com/espdev/csaps/workflows/main/badge.svg" alt="GitHub Actions (Tests)" /></a>
  <a href="https://csaps.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/csaps/badge/?version=latest" alt="Documentation Status" /></a>
  <a href="https://coveralls.io/github/espdev/csaps?branch=master"><img src="https://coveralls.io/repos/github/espdev/csaps/badge.svg?branch=master" alt="Coverage Status" /></a>
  <a href="https://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/pypi/l/csaps.svg" alt="License" /></a>
</p>

**csaps** is a Python package for univariate, multivariate and n-dimensional grid data approximation using cubic smoothing splines.
The package can be useful in practical engineering tasks for data approximation and smoothing.

## Installing

Use pip for installing:

```
pip install -U csaps
```

or Poetry:

```
poetry add csaps
```

The module depends only on NumPy and SciPy. Python 3.10 or above is supported.

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

<p align="center">
  <img src="https://user-images.githubusercontent.com/1299189/72231304-cd774380-35cb-11ea-821d-d5662cc1eedf.png" alt="univariate" />
<p/>

A surface data smoothing:

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from csaps import csaps

np.random.seed(1234)
xdata = [np.linspace(-3, 3, 41), np.linspace(-3.5, 3.5, 31)]
i, j = np.meshgrid(*xdata, indexing='ij')
ydata = (3 * (1 - j)**2. * np.exp(-(j**2) - (i + 1)**2)
         - 10 * (j / 5 - j**3 - i**5) * np.exp(-j**2 - i**2)
         - 1 / 3 * np.exp(-(j + 1)**2 - i**2))
ydata = ydata + (np.random.randn(*ydata.shape) * 0.75)

ydata_s = csaps(xdata, ydata, xdata, smooth=0.988)

fig = plt.figure(figsize=(7, 4.5))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('none')
c = [s['color'] for s in plt.rcParams['axes.prop_cycle']]
ax.plot_wireframe(j, i, ydata, linewidths=0.5, color=c[0], alpha=0.5)
ax.scatter(j, i, ydata, s=10, c=c[0], alpha=0.5)
ax.plot_surface(j, i, ydata_s, color=c[1], linewidth=0, alpha=1.0)
ax.view_init(elev=9., azim=290)

plt.show()
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/1299189/72231252-7a9d8c00-35cb-11ea-8890-487b8a7dbd1d.png" alt="surface" />
<p/>

## Documentation

More examples of usage and the full documentation can be found at https://csaps.readthedocs.io.

## Development

We use Poetry to manage the project:

```
git clone https://github.com/espdev/csaps.git
cd csaps
poetry install -E docs
```

Also, install pre-commit hooks:

```
poetry run pre-commit install
```

## Testing and Linting

We use pytest for testing and ruff/mypy for linting.
Use `poethepoet` to run tests and linters:

```
poetry run poe test
poetry run poe check
```

## Algorithm and Implementation

**csaps** Python package is inspired by MATLAB [CSAPS](https://www.mathworks.com/help/curvefit/csaps.html) function that is an implementation of 
Fortran routine SMOOTH from [PGS](http://pages.cs.wisc.edu/~deboor/pgs/) (originally written by Carl de Boor).

Also, the algothithm implementation in other languages:

* [csaps-rs](https://github.com/espdev/csaps-rs) Rust ndarray/sprs based implementation
* [csaps-cpp](https://github.com/espdev/csaps-cpp) C++11 Eigen based implementation (incomplete)

## References

C. de Boor, A Practical Guide to Splines, Springer-Verlag, 1978.

## License

[MIT](https://choosealicense.com/licenses/mit/)
