# -*- coding: utf-8 -*-

import pathlib
from setuptools import setup


NAME = 'csaps'
VERSION = '0.4.2'
ROOT_DIR = pathlib.Path(__file__).parent


def _get_long_description():
    readme = ROOT_DIR / 'README.md'
    return readme.read_text(encoding='utf-8')


setup(
    name=NAME,
    version=VERSION,
    py_modules=['csaps'],
    python_requires='>=3.5',
    install_requires=[
        'numpy>=0.12.1',
        'scipy>=0.19.1',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    url='https://github.com/espdev/csaps',
    license='MIT',
    author='Eugene Prilepin',
    author_email='esp.home@gmail.com',
    description='Cubic spline approximation (smoothing)',
    long_description=_get_long_description(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
)
