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
    CubicSmoothingSpline
    NdGridCubicSmoothingSpline

    ISplinePPForm
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

.. autoclass:: CubicSmoothingSpline
    :show-inheritance:
    :members:
    :special-members: __call__

----

.. autoclass:: NdGridCubicSmoothingSpline
    :show-inheritance:
    :members:
    :special-members: __call__

----

.. autoclass:: SplinePPForm
    :show-inheritance:
    :members:

----

.. autoclass:: NdGridSplinePPForm
    :show-inheritance:
    :members:

Interfaces
----------

.. autoclass:: ISmoothingSpline
    :show-inheritance:
    :members:
    :special-members: __call__

----

.. autoclass:: ISplinePPForm
    :show-inheritance:
    :members:
