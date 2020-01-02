.. _quickstart:

Quickstart
==========

Using csaps() function
----------------------

**csaps** provides object-oriented API for computing and evaluating univariate,
multivariate and nd-gridded splines, but in most cases we can use
a shortcut function :func:`csaps` for smoothing data and computing splines.

.. code-block::

    # Import csaps function
    from csaps import csaps

It is a simple example how to smooth univariate data:

.. plot::
    :include-source:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from csaps import csaps

    >>> np.random.seed(1234)
    >>> x = np.linspace(-5., 5., 25)
    >>> y = np.exp(-(x/2.5)**2) + (np.random.rand(25) - 0.2) * 0.3
    >>> xi = np.linspace(x[0], x[-1], 150)

    >>> yi = csaps(x, y, xi, smooth=0.85)

    >>> plt.plot(x, y, 'o', xi, yi, '-')
    >>> plt.legend(['input data', 'smoothed data'])
    >>> plt.title('Smoothing univariate data')
    >>> plt.show()


Also we can smooth multivariate data using the same function:

.. plot::
    :include-source:

    2-D data

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from csaps import csaps

    >>> np.random.seed(1234)
    >>> theta = np.linspace(0, 2*np.pi, 35)
    >>> x = np.cos(theta) + np.random.randn(35) * 0.1
    >>> y = np.sin(theta) + np.random.randn(35) * 0.1
    >>> data = [x, y]
    >>> theta_i = np.linspace(0, 2*np.pi, 200)

    >>> data_i = csaps(theta, data, theta_i, smooth=0.95)
    >>> xi = data_i[0, :]
    >>> yi = data_i[1, :]

    >>> plt.plot(x, y, ':o', xi, yi, '-')
    >>> plt.legend(['input data', 'smoothed data'])
    >>> plt.title('Smoothing 2-d data')
    >>> plt.show()

.. plot::
    :include-source:

    3-D data

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> from csaps import csaps

    >>> n = 100
    >>> theta = np.linspace(-4 * np.pi, 4 * np.pi, n)
    >>> z = np.linspace(-2, 2, n)
    >>> r = z ** 2 + 1
    >>> np.random.seed(1234)
    >>> x = r * np.sin(theta) + np.random.randn(n) * 0.3
    >>> np.random.seed(5678)
    >>> y = r * np.cos(theta) + np.random.randn(n) * 0.3
    >>> data = [x, y, z]
    >>> theta_i = np.linspace(-4 * np.pi, 4 * np.pi, 250)

    >>> data_i = csaps(theta, data, theta_i, smooth=0.95)
    >>> xi = data_i[0, :]
    >>> yi = data_i[1, :]
    >>> zi = data_i[2, :]

    >>> fig = plt.figure()
    >>> ax = fig.gca(projection='3d')
    >>> ax.plot(x, y, z, '.:', label='parametric curve')
    >>> ax.plot(xi, yi, zi, '-', label='spline curve')
    >>> plt.legend(['input data', 'smoothed data'])
    >>> plt.title('Smoothing 3-d data')
    >>> plt.show()

Finally, using the same function we can smooth nd-gridded data:

.. plot::
    :include-source:

    A surface data

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> from csaps import csaps

    >>> np.random.seed(12345)
    >>> xdata = [np.linspace(-3, 3, 41), np.linspace(-3.5, 3.5, 31)]
    >>> i, j = np.meshgrid(*xdata, indexing='ij')
    >>> ydata = (3 * (1 - j)**2. * np.exp(-(j**2) - (i + 1)**2)
    >>>          - 10 * (j / 5 - j**3 - i**5) * np.exp(-j**2 - i**2)
    >>>          - 1 / 3 * np.exp(-(j + 1)**2 - i**2))
    >>> ydata = ydata + (np.random.randn(*ydata.shape) * 0.75)

    >>> ydata_s = csaps(xdata, ydata, xdata, smooth=0.988)

    >>> fig = plt.figure(figsize=(13, 10))
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> ax.plot_wireframe(j, i, ydata, linewidths=0.5, color='r')
    >>> ax.scatter(j, i, ydata, s=10, c='r')
    >>> ax.plot_surface(j, i, ydata_s, linewidth=0, alpha=1.0)
    >>> ax.view_init(elev=9., azim=290)
    >>> plt.title('Smoothing surface data')
    >>> plt.show()

In all the examples above we used the following ``csaps`` signature::

    yi = csaps(x, y, xi, smooth)

where

    - ``x`` -- the data sites 1-d vector for univariate/multivariate cases and
      a sequence of 1-d vectors for nd-gridded case. ``x``-values must satisfy the
      condition: ``x1 < x2 < ... < xN``
    - ``y`` -- the data values. For univariate case it is vector with the same size as ``x``,
      for multivariate case it is a sequence of vectors or nd-array, and for nd-gridded data
      it is nd-array
    - ``xi`` -- the data sites for smoothed data. Usually, it in the same range as ``x``,
      but has more interpolated points
    - ``smooth`` -- the smoothing factor in the range ``[0, 1]``
