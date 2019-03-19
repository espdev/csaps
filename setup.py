# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='csaps',
    version='0.2.0',
    py_modules=['csaps'],
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
