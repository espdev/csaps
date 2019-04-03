# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='csaps',
    version='0.4.0',
    py_modules=['csaps'],
    python_requires='>=3.5',
    install_requires=[
        'numpy>=0.12.1',
        'scipy>=0.19.1',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    url='',
    license='MIT',
    author='Eugene Prilepin',
    author_email='esp.home@gmail.com',
    description='Cubic spline approximation (smoothing)'
)
