# -*- coding: utf-8 -*-

import pathlib
from setuptools import setup, find_packages


ROOT_DIR = pathlib.Path(__file__).parent


def _get_version():
    about = {}
    ver_mod = ROOT_DIR / 'csaps' / '_version.py'
    with ver_mod.open() as f:
        exec(f.read(), about)
    return about['__version__']


def _get_long_description():
    readme = ROOT_DIR / 'README.md'
    return readme.read_text(encoding='utf-8')


setup(
    name='csaps',
    version=_get_version(),
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    python_requires='>=3.5',
    install_requires=[
        'numpy>=0.12.1',
        'scipy>=0.19.1',
    ],
    package_data={"csaps": ["py.typed"]},
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
