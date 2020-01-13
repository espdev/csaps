.. _api:

API Reference
=============

Summary
-------

.. currentmodule:: csaps

.. autosummary::
    :nosignatures:

    csaps
    AutoSmoothingResult

    ISmoothingSpline
    UnivariateCubicSmoothingSpline
    MultivariateCubicSmoothingSpline
    NdGridCubicSmoothingSpline

    SplinePPFormBase
    SplinePPForm
    NdGridSplinePPForm

Main API
--------

.. py:module:: csaps

.. autofunction:: csaps

----

.. autoclass:: AutoSmoothingResult
    :show-inheritance:
    :members:

Object-Oriented API
-------------------

.. autoclass:: UnivariateCubicSmoothingSpline
    :members:
    :special-members: __call__

----

.. autoclass:: MultivariateCubicSmoothingSpline
    :members:
    :special-members: __call__

----

.. autoclass:: NdGridCubicSmoothingSpline
    :members:
    :special-members: __call__

----

.. autoclass:: SplinePPForm
    :members:

----

.. autoclass:: NdGridSplinePPForm
    :members:

Base Classes
------------

.. autoclass:: ISmoothingSpline
    :members:
    :special-members: __call__

----

.. autoclass:: SplinePPFormBase
    :members:
