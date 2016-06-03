#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import shutil
from setuptools import setup


pkg_name = 'pyneqsys'

RELEASE_VERSION = os.environ.get('PYNEQSYS_RELEASE_VERSION', '')

# http://conda.pydata.org/docs/build.html#environment-variables-set-during-the-build-process
if os.environ.get('CONDA_BUILD', '0') == '1':
    try:
        RELEASE_VERSION = 'v' + io.open('__conda_version__.txt', 'rt',
                                        encoding='utf-8').readline().rstrip()
    except IOError:
        pass


def _path_under_setup(*args):
    return os.path.join(os.path.dirname(__file__), *args)

release_py_path = _path_under_setup(pkg_name, '_release.py')

if len(RELEASE_VERSION) > 1 and RELEASE_VERSION[0] == 'v':
    TAGGED_RELEASE = True
    __version__ = RELEASE_VERSION[1:]
else:
    TAGGED_RELEASE = False
    # read __version__ attribute from _release.py:
    exec(io.open(release_py_path, encoding='utf-8').read())

classifiers = [
    "Development Status :: 4 - Beta",
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
]

tests = [
    'pyneqsys.tests',
]

with io.open(_path_under_setup(pkg_name, '__init__.py'),
             encoding='utf-8') as f:
    long_description = f.read().split('"""')[1]

descr = 'Solving of symbolic systems of non-linear equations numerically.'
setup_kwargs = dict(
    name=pkg_name,
    version=__version__,
    description=descr,
    long_description=long_description,
    classifiers=classifiers,
    author='Björn Dahlgren',
    author_email='bjodah@DELETEMEgmail.com',
    url='https://github.com/bjodah/' + pkg_name,
    license='BSD',
    packages=[pkg_name] + tests,
    install_requires=['numpy'],
    extras_require={'all': ['sympy', 'scipy', 'pyodesys']}
)

if __name__ == '__main__':
    try:
        if TAGGED_RELEASE:
            # Same commit should generate different sdist
            # depending on tagged version (set PYNEQSYS_RELEASE_VERSION)
            # this will ensure source distributions contain the correct version
            shutil.move(release_py_path, release_py_path+'__temp__')
            open(release_py_path, 'wt').write(
                "__version__ = '{}'\n".format(__version__))
        setup(**setup_kwargs)
    finally:
        if TAGGED_RELEASE:
            shutil.move(release_py_path+'__temp__', release_py_path)
