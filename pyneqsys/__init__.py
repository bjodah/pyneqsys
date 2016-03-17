# -*- coding: utf-8 -*-
"""
Package for solving of symbolic systems of non-linear equations numerically.

pyneqsys ties computer algebra systems like:

- SymPy
- symengine
- symcxx

and numerical solvers such as:

- MINPACK in SciPy
- NLEQ2 in pynleq2
- KINSOL in pykinsol

in addition pyneqsys provides abstraction classes for e.g. having
conditional equations in a system.
"""

from __future__ import absolute_import

from ._release import __version__
from .core import NeqSys, ConditionalNeqSys, ChainedNeqSys
