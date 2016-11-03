# -*- coding: utf-8 -*-
"""
Package for numerically solving symbolically defined systems of non-linear equations.

pyneqsys ties computer algebra systems like:

- SymPy
- symengine
- SymCXX

and numerical solvers such as:

- MINPACK in SciPy
- NLEQ2 in pynleq2
- KINSOL in pykinsol

together. In addition ``pyneqsys`` provides helper classes for
handling e.g. conditional equations in a system.
"""

from __future__ import absolute_import

from ._release import __version__
from .core import NeqSys, ConditionalNeqSys, ChainedNeqSys
