from __future__ import print_function, absolute_import, division

from .. import SymbolicSys
from .test_core import mk_f


def test_symbolicsys__from_callback():
    ss = SymbolicSys.from_callback(mk_f(3), 2)
    sol = ss.solve_scipy([0, 0])
    assert abs(sol.x[0] - 0.8411639) < 2e-7
    assert abs(sol.x[1] - 0.1588361) < 2e-7
