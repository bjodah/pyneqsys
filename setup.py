#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
from itertools import chain
import os
import re
import shutil
import subprocess
import sys
import warnings
from setuptools import setup

pkg_name = 'pyneqsys'
url = 'https://github.com/bjodah/' + pkg_name
license = 'BSD'

RELEASE_VERSION = os.environ.get('%s_RELEASE_VERSION' % pkg_name.upper(), '')  # v*


def _path_under_setup(*args):
    return os.path.join(os.path.dirname(__file__), *args)

release_py_path = _path_under_setup(pkg_name, '_release.py')

if len(RELEASE_VERSION) > 0:
    if RELEASE_VERSION[0] == 'v':
        TAGGED_RELEASE = True
        __version__ = RELEASE_VERSION[1:]
    else:
        raise ValueError("Ill formated version")
else:
    TAGGED_RELEASE = False
    # read __version__ attribute from _release.py:
    exec(io.open(release_py_path, encoding='utf-8').read())
    if __version__.endswith('git'):
        try:
            _git_version = subprocess.check_output(
                ['git', 'describe', '--dirty']).rstrip().decode('utf-8').replace('-dirty', '.dirty')
        except subprocess.CalledProcessError:
            warnings.warn("A git-archive is being installed - version information incomplete.")
        else:
            if 'develop' not in sys.argv:
                warnings.warn("Using git to derive version: dev-branches may compete.")
                _ver_tmplt = r'\1.post\2' if os.environ.get('CONDA_BUILD', '0') == '1' else r'\1.post\2+\3'
                __version__ = re.sub('v([0-9.]+)-(\d+)-(\S+)', _ver_tmplt, _git_version)  # .dev < '' < .post

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

with io.open(_path_under_setup(pkg_name, '__init__.py'), encoding='utf-8') as f:
    short_description = f.read().split('"""')[1].split('\n')[1]
if not 10 < len(short_description) < 255:
    warnings.warn("Short description from __init__.py proably not read correctly")
long_descr = io.open(_path_under_setup('README.rst'), encoding='utf-8').read()
if not len(long_descr) > 100:
    warnings.warn("Long description from README.rst probably not read correctly.")
_author, _author_email = open(_path_under_setup('AUTHORS'), 'rt').readline().split('<')

extras_req = {
    'symbolic': ['pysym', 'symcxx'],  # use conda for symengine
    'docs': ['Sphinx', 'sphinx_rtd_theme', 'numpydoc'],
    'solvers': ['pykinsol', 'levmar', 'mpmath'],  # maybe also cyipopt and pynleq2
    'testing': ['pytest', 'pytest-cov', 'pytest-flakes', 'pytest-pep8']
}
extras_req['all'] = list(chain(extras_req.values()))

setup_kwargs = dict(
    name=pkg_name,
    version=__version__,
    description=short_description,
    long_description=long_descr,
    classifiers=classifiers,
    author=_author,
    author_email=_author_email.split('>')[0].strip(),
    url=url,
    license=license,
    packages=[pkg_name] + tests,
    install_requires=['numpy>1.7', 'sym>=0.3.1', 'sympy>=1.3', 'scipy', 'matplotlib'],
    extras_require=extras_req
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
