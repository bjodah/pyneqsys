from __future__ import (absolute_import, division, print_function)

from .. import NeqSys


def mk_f(n):
    # docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html
    def f(x):
        return [x[0] + (x[0] - x[1])**n/2 - 1,
                (x[1] - x[0])**n/2 + x[1]]

    return f


def mk_j(n):
    def j(x):
        return [
            [
                1 + n/2 * (x[0] - x[1])**(n-1),
                -n/2 * (x[0] - x[1])**(n-1)
            ],
            [
                -n/2 * (x[1] - x[0])**(n-1),
                1 + n/2 * (x[1] - x[0])**(n - 1)
            ]
        ]

    return j


def test_neqsys__solve_scipy():
    ns = NeqSys(2, 2, mk_f(3), jac=mk_j(3))
    x, sol = ns.solve_scipy([0, 0])
    assert abs(x[0] - 0.8411639) < 2e-7
    assert abs(x[1] - 0.1588361) < 2e-7


def test_neqsys_params():
    def f(x, n):
        return mk_f(n)(x)

    def j(x, n):
        return mk_j(n)(x)

    ns = NeqSys(2, 2, f, jac=j)
    x, sol = ns.solve_scipy([0, 0], 3)
    assert abs(x[0] - 0.8411639) < 2e-7
    assert abs(x[1] - 0.1588361) < 2e-7
