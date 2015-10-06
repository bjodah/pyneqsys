from __future__ import print_function, absolute_import, division

import sympy as sp

from .. import SymbolicSys
from ..symbolic import linear_exprs, TransformedSys
from .test_core import mk_f


def test_symbolicsys__from_callback():
    ss = SymbolicSys.from_callback(mk_f(3), 2)
    x, sol = ss.solve_scipy([0, 0])
    assert sol.success
    assert abs(x[0] - 0.8411639) < 2e-7
    assert abs(x[1] - 0.1588361) < 2e-7


def test_symbolicsys__from_callback__params():
    def f(x, args):
        n = args[0]
        return mk_f(n)(x)

    ss = SymbolicSys.from_callback(f, 2, 1)
    x, sol = ss.solve_scipy([.7, .3], 3)
    assert sol.success
    assert abs(x[0] - 0.8411639) < 2e-7
    assert abs(x[1] - 0.1588361) < 2e-7


def test_TransformedSys__from_callback():
    ts = TransformedSys.from_callback(mk_f(3), 2, (sp.exp, sp.log))
    x, sol = ts.solve('scipy', [1, 1])
    assert sol.success
    print(sol)
    assert abs(x[0] - 0.8411639) < 2e-7
    assert abs(x[1] - 0.1588361) < 2e-7


def test_linear_exprs():
    a, b, c = x = sp.symarray('x', 3)
    coeffs = [[1, 3, -2],
              [3, 5, 6],
              [2, 4, 3]]
    vals = [5, 7, 8]
    exprs = linear_exprs(x, coeffs, vals)
    known = [1*a + 3*b - 2*c - 5,
             3*a + 5*b + 6*c - 7,
             2*a + 4*b + 3*c - 8]
    assert all([(rt - kn).simplify() == 0 for rt, kn in zip(exprs, known)])

    rexprs = linear_exprs(x, coeffs, vals, rref=True)
    rknown = [a + 15, b - 8, c - 2]
    assert all([(rt - kn).simplify() == 0 for rt, kn in zip(rexprs, rknown)])
