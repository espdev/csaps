.. _manual:

Manual
======


.. _manual-quickstart:

Quickstart
----------

**csaps** provides object-oriented API for computing and evaluating univariate,
multivariate and nd-gridded splines, but in most cases we recommend to use
a shortcut function :func:`csaps` for smoothing data and computing splines.

Firstly, we import :func:`csaps` function (and other modules for our examples)

.. jupyter-execute::

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    from csaps import csaps


Univariate smoothing
~~~~~~~~~~~~~~~~~~~~

Univariate data are two vectors: X and Y with the same size. X is data sites, Y is data values.
X-values must satisfy the condition: :math:`x1 < x2 < ... < xN`

It is a simple example how to smooth univariate data:

.. jupyter-execute::

    def univariate_data(n=25, seed=1234):
        np.random.seed(seed)
        x = np.linspace(-5., 5., n)
        y = np.exp(-(x/2.5)**2) + (np.random.rand(n) - 0.2) * 0.3
        return x, y

    x, y = univariate_data()
    xi = np.linspace(x[0], x[-1], 150)

    yi = csaps(x, y, xi, smooth=0.85)

    plt.plot(x, y, 'o', xi, yi, '-')
    plt.legend(['input data', 'smoothed data'])
    plt.title('Smoothing univariate data')
    plt.show()


Multivariate smoothing
~~~~~~~~~~~~~~~~~~~~~~

Also we can smooth multivariate (n-dimensional) data using the same function.

The algorithm supports computing and evaluating spline for multivariate data with vectorization.
You can compute smoothing splines for X, Y data where X is data site vector and Y is
ND-array of data value vectors.

The example of data:

.. code-block:: python

    # data sites
    x = [1, 2, 3, 4]

    # 3 data vectors (3-D)
    y = [
        (2, 4, 6, 8),
        (1, 3, 5, 7),
        (5, 1, 3, 9),
    ]

By default, the shape of Y array must be: ``(d0, d1, ..., dN)``
where ``dN`` must equal to X vector size. Also you can use ``axis`` parameter to
set the data values axis for Y array.

In this case the smoothing spline will be computed for all Y data vectors at a time.
The same weights vector and the same smoothing parameter will be used for all Y data.

**2-D data**

.. jupyter-execute::

    np.random.seed(1234)
    theta = np.linspace(0, 2*np.pi, 35)
    x = np.cos(theta) + np.random.randn(35) * 0.1
    y = np.sin(theta) + np.random.randn(35) * 0.1
    data = [x, y]
    theta_i = np.linspace(0, 2*np.pi, 200)

    data_i = csaps(theta, data, theta_i, smooth=0.95)
    xi = data_i[0, :]
    yi = data_i[1, :]

    plt.plot(x, y, ':o', xi, yi, '-')
    plt.legend(['input data', 'smoothed data'])
    plt.title('Smoothing 2-d data')
    plt.show()

**3-D data**

.. jupyter-execute::

    np.random.seed(1234)
    n = 100
    theta = np.linspace(-4 * np.pi, 4 * np.pi, n)
    z = np.linspace(-2, 2, n)
    r = z ** 2 + 1
    x = r * np.sin(theta) + np.random.randn(n) * 0.3
    y = r * np.cos(theta) + np.random.randn(n) * 0.3
    data = [x, y, z]
    theta_i = np.linspace(-4 * np.pi, 4 * np.pi, 250)

    data_i = csaps(theta, data, theta_i, smooth=0.95)
    xi = data_i[0, :]
    yi = data_i[1, :]
    zi = data_i[2, :]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, '.:', label='parametric curve')
    ax.plot(xi, yi, zi, '-', label='spline curve')
    plt.legend(['input data', 'smoothed data'])
    plt.title('Smoothing 3-d data')
    plt.show()


ND-grid smoothing
~~~~~~~~~~~~~~~~~

Finally, using the same function we can smooth nd-gridded data.

The algorithm can make smoothing splines for ND-gridded data smoothing.
In this case the algorithm makes coordinatewise smoothing (tensor-product of univariate splines coefficients).

X-data must be a sequence of vectors for each dimension. Y-data must be ND-array.

The example of data:

.. code-block:: python

    x = [
        (-2, -1, 0, 1, 2),    # X-grid data sites
        (-2, -1, 0, 1, 2),    # Y-grid data sites
        (-2, -1, 0, 1, 2),    # Z-grid data sites
    ]

    y = np.random.rand(5, 5, 5)  # 5x5x5 3-D grid data values

Also you can set the smoothing parameter for each dimension:

.. code-block:: python

    smooth = [
        0.95,  # the smoothing parameter for X
        0.83,  # the smoothing parameter for Y
        0.51,  # the smoothing parameter for Z
    ]

**A surface data**

.. jupyter-execute::

    np.random.seed(1234)
    xdata = [np.linspace(-3, 3, 41), np.linspace(-3.5, 3.5, 31)]
    i, j = np.meshgrid(*xdata, indexing='ij')
    ydata = (3 * (1 - j)**2. * np.exp(-(j**2) - (i + 1)**2)
             - 10 * (j / 5 - j**3 - i**5) * np.exp(-j**2 - i**2)
             - 1 / 3 * np.exp(-(j + 1)**2 - i**2))
    ydata = ydata + (np.random.randn(*ydata.shape) * 0.75)

    ydata_s = csaps(xdata, ydata, xdata, smooth=0.988)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(j, i, ydata, linewidths=0.5, color='r', alpha=0.5)
    ax.scatter(j, i, ydata, s=10, c='r', alpha=0.5)
    ax.plot_surface(j, i, ydata_s, linewidth=0, alpha=1.0)
    ax.view_init(elev=9., azim=290)
    plt.title('Smoothing surface data')
    plt.show()


Summary
~~~~~~~

In all the smoothing examples above we are used the following :func:`csaps` signature::

    yi = csaps(x, y, xi, smooth)

where

    - ``x`` -- the data sites vector for univariate/multivariate data and
      a sequence of vectors for nd-gridded data. ``x``-values **must** satisfy the
      condition: ``x1 < x2 < ... < xN``
    - ``y`` -- the data values. For univariate case it is vector with the same size as ``x``,
      for multivariate case it is a sequence of vectors or nd-array, and for nd-gridded data
      it is nd-array
    - ``xi`` -- the data sites for smoothed data. It is shape-like ``x`` data and in the same
      range as ``x``, but usually has more interpolated points
    - ``smooth`` -- the smoothing parameter in the range ``[0, 1]``


.. _manual-advanced:

Advanced usage
--------------


Automatic smoothing
~~~~~~~~~~~~~~~~~~~

If we want to smooth the data without specifying the smoothing parameter we can use the following
signature::

    yi, smooth = csaps(x, y, xi)

In this case the smoothing parameter will be computed automatically and will be returned in the
function result. In this case the function will return `SmoothingResult` named tuple: ``SmoothingResult(values, smooth)``.

.. jupyter-execute::

    x, y = univariate_data()
    xi = np.linspace(x[0], x[-1], 51)

    smoothing_result = csaps(x, y, xi)
    yi = smoothing_result.values

    print('Computed smoothing parameter:', smoothing_result.smooth)

    plt.plot(x, y, 'o', xi, yi, '-')
    plt.show()

In ND-gridded data case we can use auto smoothing for all dimensions or the particular dimensions:

.. code-block:: python

    smooth = [
        0.95,
        None,  # auto smoothing only for Y
        0.85,
    ]

    ...

    smoothing_result = csaps(x, y, xi, smooth=smooth)


Weighted smoothing
~~~~~~~~~~~~~~~~~~

If we want to use error measure weights while computing spline,
we can use the following signatures::

    yi = csaps(x, y, xi, weights, smooth)
    yi, smooth = csaps(x, y, xi, weights)
    spline = csaps(x, y, weights)
    spline = csaps(x, y, weights, smooth)

For example:

.. jupyter-execute::

    x, y = univariate_data()
    xi = np.linspace(x[0], x[-1], 150)

    w = np.ones_like(x) * 0.5
    w[-7:] = 0.1
    w[:7] = 0.1
    w[[10,13]] = 1.0
    w[[11,12]] = 0.1

    print('Weights:', w)

    yi = csaps(x, y, xi, smooth=0.85)
    yi_w = csaps(x, y, xi, weights=w, smooth=0.85)

    plt.plot(x, y, 'o', xi, yi, '-', xi, yi_w, '-')
    plt.legend(['input data', 'smoothed data', 'weighted smoothed data'])
    plt.show()

In ND-gridded data case we can use the same weights for all dimensions or different
weights for each dimension.


Using axis parameter
~~~~~~~~~~~~~~~~~~~~

**axis** parameter specifies :math:`y`-data axis for computing spline in multivariate/vectorize data cases.
By default axis is equal to -1 (the last axis). In other words, ``y.shape[axis]`` must be equal to ``x.size``.

The following example will raise ``ValueError``:

.. jupyter-execute::
    :raises: ValueError

    x, y1 = univariate_data(seed=1327)
    x, y2 = univariate_data(seed=2451)

    # We stack y-data as MxN array
    y = np.stack((y1, y2), axis=1)

    print('x.size:', x.size)
    print('y.shape:', y.shape)

    xi = np.linspace(x[0], x[-1], 150)
    yi = csaps(x, y, xi, smooth=0.8)

We can use ``axis`` parameter ``==0`` to fix it:

.. jupyter-execute::

    yi = csaps(x, y, xi, smooth=0.8, axis=0)

    plt.plot(x, y, 'o', xi, yi, '-')
    plt.show()

.. note::

    ``axis`` parameter is ignored in ND-gridded data cases.


Computing spline without evaluating
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we want to compute spline only without evaluating (smoothing data), we can use the following signatures::

    spline = csaps(x, y)
    spline = csaps(x, y, smooth)

In this case the smoothing spline will be computed and returned.

.. jupyter-execute::

    x, y = univariate_data(n=11)

    spline = csaps(x, y)

    print('Spline class name:', type(spline).__name__)
    print('Spline smoothing parameter:', spline.smooth)
    print('Spline description:', spline.spline)

Now we can use the computed spline to evaluate (smoothing) data for given data sites repeatedly:

.. jupyter-execute::

    xi1 = np.linspace(x[0], x[-1], 20)
    xi2 = np.linspace(x[0], x[-1], 50)

    yi1 = spline(xi1)
    yi2 = spline(xi2)

    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(x, y, 's', xi1, yi1, 'o-')
    ax2.plot(x, y, 's', xi2, yi2, 'o-')
    plt.show()
