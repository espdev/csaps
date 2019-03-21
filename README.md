# CSAPS: Cubic spline approximation (smoothing)

This package provides cubic smoothing spline for data approximation.
The smoothing parameter can be calculated automatically or it can be set manually. 

The smoothing parameter should be in range `[0, 1]` where:
* 0: The smoothing spline is the least-squares straight line fit to the data
* 1: The natural cubic spline interpolant

The calculation of the smoothing spline requires the solution of a linear system 
whose coefficient matrix has the form `p*A + (1 - p)*B`, with the matrices `A` and `B` 
depending on the data sites `X`. The automatically computed smoothing parameter makes 
`p*trace(A) equal (1 - p)*trace(B)`.

## The examples of usage

### Smoothing univariate data

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

### Weighted smoothing univariate data

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
The shape of `Y` array must be: `(s1, s2, ..., sN)` where `sN` must equal of `X` vector size.

In this case the smoothing splines will be computed for all `Y` data vectors at a time.

For example:

```python
import numpy
import csaps

# data sites
x = [1, 2, 3, 4]

# two data vectors
y = [(2, 4, 6, 8), 
     (1, 3, 5, 7)]

sp = csaps.UnivariateCubicSmoothingSpline(x, y)

xi = numpy.linspace(1, 4, 10)
yi = sp(xi)

print(yi.shape)  # (2, 10)
assert yi.shape[:-1] == yi.shape[:-1]
```

**Important**:
The same weights vector and the same smoothing parameter will be used for all Y data.

## Algorithms and implementations

`csaps` is a Python/NumPy rough port of MATLAB `csaps` function that is an implementation of 
Fortran routine SMOOTH from PGS.

[csaps-cpp](https://github.com/espdev/csaps-cpp) C++11 Eigen based implementation of the algorithm.

## References

C. de Boor, A Practical Guide to Splines, Springer-Verlag, 1978.
