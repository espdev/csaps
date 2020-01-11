# -*- coding: utf-8 -*-

import pathlib
from setuptools import setup


ROOT_DIR = pathlib.Path(__file__).parent


def _get_version():
    about = {}
    ver_mod = ROOT_DIR / 'csaps' / '_version.py'
    exec(ver_mod.read_text(), about)
    return about['__version__']


def _get_long_description():
    readme = ROOT_DIR / 'README.md'
    return readme.read_text(encoding='utf-8')


setup(
    name='csaps',
    version=_get_version(),
    packages=['csaps'],
    python_requires='>=3.5, <4',
    install_requires=[
        'numpy >=0.12.1, <1.20.0',
        'scipy >=0.19.1, <1.6.0',
    ],
    extras_require={
        'docs': [
            'sphinx >=2.3',
            'matplotlib >=3.1',
            'jupyter-sphinx',
            'numpydoc',
            'm2r',
        ],
        'tests': [
            'pytest',
        ],
    },
    package_data={"csaps": ["py.typed"]},
    url='https://github.com/espdev/csaps',
    project_urls={
        'Documentation': 'https://csaps.readthedocs.io',
        'Code': 'https://github.com/espdev/csaps',
        'Issue tracker': 'https://github.com/espdev/csaps/issues',
    },
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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
)
