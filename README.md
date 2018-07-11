# CSAPS: Cubic spline approximation (smoothing)

## Usage

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

## References

C. de Boor, A Practical Guide to Splines, Springer-Verlag, 1978.
