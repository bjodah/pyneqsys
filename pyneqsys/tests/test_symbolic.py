from __future__ import print_function, absolute_import, division

import pytest

try:
    import sympy as sp
    from ..symbolic import SymbolicSys, linear_exprs, TransformedSys
except ImportError:
    missing_import = True
else:
    missing_import = False

from .test_core import f


@pytest.mark.skipif(missing_import, reason="pyneqsys.symbolic req. missing")
def test_SymbolicSys():
    # from: http://stackoverflow.com/questions/33135238
    a, b, t = sp.symbols('a b t')

    def f(x):
        return 1/(x+a)**t + b
    neqsys = SymbolicSys([a, b], [f(0) - 1, f(1) - 0], [t])
    ab, sol = neqsys.solve([0.5, -0.5], 1, solver='scipy')
    assert sol['success']
    assert abs(ab[0] - (-1/2 + 5**0.5/2)) < 1e-10
    assert abs(ab[1] - (1/2 - 5**0.5/2)) < 1e-10


@pytest.mark.skipif(missing_import, reason="pyneqsys.symbolic req. missing")
def test_symbolicsys__from_callback():
    ss = SymbolicSys.from_callback(f, 2, 1)
    x, sol = ss.solve([1, 0], [3], solver='scipy')
    assert sol['success']
    assert abs(x[0] - 0.8411639) < 2e-7
    assert abs(x[1] - 0.1588361) < 2e-7


@pytest.mark.skipif(missing_import, reason="pyneqsys.symbolic req. missing")
def test_symbolicsys__from_callback__no_params():
    def _nf(x):
        return f(x, [3])

    ss = SymbolicSys.from_callback(_nf, 2)
    x, sol = ss.solve([.7, .3], solver='scipy')
    assert sol['success']
    assert abs(x[0] - 0.8411639) < 2e-7
    assert abs(x[1] - 0.1588361) < 2e-7


@pytest.mark.skipif(missing_import, reason="pyneqsys.symbolic req. missing")
def test_TransformedSys__from_callback():
    ts = TransformedSys.from_callback(f, (sp.exp, sp.log), 2, 1)
    x, sol = ts.solve([1, .1], [3], solver='scipy')
    assert sol['success']
    assert abs(x[0] - 0.8411639) < 2e-7
    assert abs(x[1] - 0.1588361) < 2e-7


@pytest.mark.skipif(missing_import, reason="pyneqsys.symbolic req. missing")
def test_SymbolicSys__from_callback__method():

    class Problem(object):
        def f(self, x, p):
            return [x[0]**p[0]]

    p = Problem()
    ss = SymbolicSys.from_callback(p.f, 1, 1)
    x, sol = ss.solve([1], [3])
    assert abs(x[0]) < 1e-14


@pytest.mark.skipif(missing_import, reason="pyneqsys.symbolic req. missing")
def test_linear_exprs():
    a, b, c = x = sp.symarray('x', 3)
    coeffs = [[1, 3, -2],
              [3, 5, 6],
              [2, 4, 3]]
    vals = [5, 7, 8]
    exprs = linear_exprs(coeffs, x, vals)
    known = [1*a + 3*b - 2*c - 5,
             3*a + 5*b + 6*c - 7,
             2*a + 4*b + 3*c - 8]
    assert all([(rt - kn).simplify() == 0 for rt, kn in zip(exprs, known)])

    rexprs = linear_exprs(coeffs, x, vals, rref=True)
    rknown = [a + 15, b - 8, c - 2]
    assert all([(rt - kn).simplify() == 0 for rt, kn in zip(rexprs, rknown)])
