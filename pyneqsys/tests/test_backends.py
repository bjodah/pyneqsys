# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import pytest


def _test(**kwargs):
    from ..symbolic import SymbolicSys
    from .test_core import f

    ss = SymbolicSys.from_callback(
        f, 2, 1, **kwargs)

    x, sol = ss.solve([1, 0], [3], solver='scipy')
    assert sol['success']
    assert abs(x[0] - 0.8411639) < 2e-7
    assert abs(x[1] - 0.1588361) < 2e-7


try:
    import pysym
except ImportError:
    pysym = None


@pytest.mark.skipif(pysym is None, reason="pysym missing")
def test_pysym_SymbolicSys_from_callback():
    _test(backend='pysym')


try:
    import symcxx
except ImportError:
    symcxx = None


@pytest.mark.skipif(symcxx is None, reason="symcxx missing")
def test_symcxx_SymbolicSys_from_callback():
    _test(backend='symcxx')


try:
    import symengine
except ImportError:
    symengine = None


@pytest.mark.skipif(symengine is None, reason="symengine missing")
def test_symengine_SymbolicSys_from_callback():
    _test(backend='symengine')
