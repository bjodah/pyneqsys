# -*- coding: utf-8 -*-
"""
Package for solving of symbolic systems of non-linear equations numerically.

pyneqsys ties computer algebra systems like SymPy and symengine, and numerical
solvers such as MINPACK in SciPy or NLEQ2 in pynleq2 together.
"""

from __future__ import absolute_import

from .core import NeqSys, ConditionalNeqSys
from .symbolic import SymbolicSys

assert NeqSys  # silence pyflakes
assert SymbolicSys  # silence pyflakes
assert ConditionalNeqSys  # silence pyflakes
