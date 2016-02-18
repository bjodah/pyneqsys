# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import pytest

try:
    import pysym
except ImportError:
    pysym = None
else:
    from ..symbolic import SymbolicSys
    from .test_core import f


@pytest.mark.skipif(pysym is None, reason="pysym missing")
def test_pysym_SymbolicSys_from_callback():
    ss = SymbolicSys.from_callback(
        f, 2, 1,
        lambdify=pysym.Lambdify,
        lambdify_unpack=False,
        symarray=pysym.symarray,
        Matrix=pysym.Matrix)

    x, sol = ss.solve([1, 0], [3], solver='scipy')
    assert sol['success']
    assert abs(x[0] - 0.8411639) < 2e-7
    assert abs(x[1] - 0.1588361) < 2e-7
