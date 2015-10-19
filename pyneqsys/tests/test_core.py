from __future__ import (absolute_import, division, print_function)

from .. import NeqSys


def f(x, params):
    # docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html
    return [x[0] + (x[0] - x[1])**params[0]/2 - 1,
            (x[1] - x[0])**params[0]/2 + x[1]]


def j(x, params):
    return [
        [
            1 + params[0]/2 * (x[0] - x[1])**(params[0]-1),
            -params[0]/2 * (x[0] - x[1])**(params[0]-1)
        ],
        [
            -params[0]/2 * (x[1] - x[0])**(params[0]-1),
            1 + params[0]/2 * (x[1] - x[0])**(params[0] - 1)
        ]
    ]


def test_neqsys_params():
    ns = NeqSys(2, 2, f, jac=j)
    x, sol = ns.solve_scipy([0, 0], [3])
    assert abs(x[0] - 0.8411639) < 2e-7
    assert abs(x[1] - 0.1588361) < 2e-7


def test_neqsys_no_params():
    ns = NeqSys(2, 2, lambda x: f(x, [3]),
                jac=lambda x: j(x, [3]))
    x, sol = ns.solve_scipy([0, 0])
    assert abs(x[0] - 0.8411639) < 2e-7
    assert abs(x[1] - 0.1588361) < 2e-7
