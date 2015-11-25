# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import numpy as np
from .. import NeqSys, ConditionalNeqSys


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


def _test_neqsys_params(solver):
    ns = NeqSys(2, 2, f, jac=j)
    x, sol = ns.solve(solver, [0, 0], [3])
    assert abs(x[0] - 0.8411639) < 2e-7
    assert abs(x[1] - 0.1588361) < 2e-7


def _test_neqsys_solve_series(solver):
    ns = NeqSys(2, 2, f, jac=j)
    x, sol = ns.solve_series(solver, [0, 0], [0], var_data=[2, 3], var_idx=0)
    assert abs(x[0, 0] - 0.5) < 2e-7
    assert abs(x[0, 1] + 0.5) < 2e-7
    assert abs(x[1, 0] - 0.8411639) < 2e-7
    assert abs(x[1, 1] - 0.1588361) < 2e-7


def test_neqsys_params_scipy():
    _test_neqsys_params('scipy')


def test_neqsys_params_nleq2():
    _test_neqsys_params('nleq2')


def _test_neqsys_no_params(solver):
    ns = NeqSys(2, 2, lambda x: f(x, [3]),
                jac=lambda x: j(x, [3]))
    x, sol = ns.solve(solver, [0, 0])
    assert abs(x[0] - 0.8411639) < 2e-7
    assert abs(x[1] - 0.1588361) < 2e-7


def test_neqsys_no_params_scipy():
    _test_neqsys_no_params('scipy')


def test_neqsys_no_params_nleq2():
    _test_neqsys_no_params('nleq2')


def test_ConditionalNeqSys1():
    from math import pi, sin

    def f_a(x, p):
        return [sin(p[0]*x[0])]  # when x <= 0

    def f_b(x, p):
        return [x[0]*(p[1]-x[0])]  # when x >= 0

    def factory(conds):
        return NeqSys(1, 1, f_b) if conds[0] else NeqSys(1, 1, f_a)

    cneqsys = ConditionalNeqSys([
        (lambda x, p: x[0] > 0, lambda x, p: x[0] >= 0)], factory)
    x, sol = cneqsys.solve('scipy', [0], [pi, 3])
    assert sol.success
    assert abs(x[0]) < 1e-13
    x, sol = cneqsys.solve('scipy', [-1.4], [pi, 3])
    assert sol.success
    assert abs(x[0] + 1) < 1e-13
    x, sol = cneqsys.solve('scipy', [2], [pi, 3])
    assert sol.success
    assert abs(x[0] - 3) < 1e-13
    x, sol = cneqsys.solve('scipy', [7], [pi, 3])
    assert sol.success
    assert abs(x[0] - 3) < 1e-13


def _check_NaCl(cneqsys, guesses, cases=-1, **kwargs):
    _init_final = [
        ([1, 1, 1], [2, 2, 0]),
        ([1, 1, 0], [1, 1, 0]),
        ([3, 3, 3], [2, 2, 4]),
        ([2, 2, 0], [2, 2, 0]),
        ([2+1e-8, 2+1e-8, 0], [2, 2, 1e-8]),
        ([3, 3, 0], [2, 2, 1]),
        ([0, 0, 3], [2, 2, 1]),
        ([0, 0, 2], [2, 2, 0]),
        ([2, 2, 2], [2, 2, 2]),
    ]

    for init, final in _init_final[:cases]:
        print(init)
        for guess in guesses:
            print(guess)
            if guess is None:
                guess = init
            x, sol = cneqsys.solve('scipy', guess, init + [4], **kwargs)
            assert sol.success and np.allclose(x, final)


def test_ConditionalNeqSys2():
    # This is an example of NaCl precipitation
    # x = Na+, Cl-, NaCl(s)
    # p = [Na+]0, [Cl-]0, [NaCl(s)]0, Ksp
    # f[0] = x[0] + x[2] - p[0] - p[2]
    # f[1] = x[1] + x[2] - p[1] - p[2]
    # switch to precipitation: x[0]*x[1] > p[3]
    # keep precipitation if: x[2] > 0
    #
    # If we have a precipitate
    #    f[2] = x[0]*x[1] - p[3]
    # otherwise:
    #    f[2] = x[2]

    def factory(conds):
        precip = conds[0]

        def cb(x, p):
            f = [None]*3
            f[0] = x[0] + x[2] - p[0] - p[2]
            f[1] = x[1] + x[2] - p[1] - p[2]
            if precip:
                f[2] = x[0]*x[1] - p[3]
            else:
                f[2] = x[2]
            return f
        return NeqSys(3, 3, cb)

    cneqsys = ConditionalNeqSys([
        (lambda x, p: x[0]*x[1] > p[3],
         lambda x, p: x[2] > 0)
    ], factory)

    _check_NaCl(cneqsys, [(1, 1, 1), (1, 1, 0), (2, 2, 0), (1, 1, 3)])


def test_ConditionalNeqSys3():
    # This is equivalent to ConditionalNeqSys3
    # but this time variable transformations
    # are performed
    from math import exp, log

    small = -60

    def pre_processor(x, p):
        return np.log(np.asarray(x) + exp(small)), p

    def post_processor(x, p):
        return np.exp(x), p

    def factory(conds):
        precip = conds[0]

        def fun(x, p):
            f = [None]*3
            f[0] = exp(x[0]) + exp(x[2]) - p[0] - p[2]
            f[1] = exp(x[1]) + exp(x[2]) - p[1] - p[2]
            if precip:
                f[2] = x[0] + x[1] - log(p[3])
            else:
                f[2] = x[2] - small
            return f

        def jac(x, p):
            jout = np.empty((3, 3))

            jout[0, 0] = exp(x[0])
            jout[0, 1] = 0
            jout[0, 2] = exp(x[2])

            jout[1, 0] = 0
            jout[1, 1] = exp(x[1])
            jout[1, 2] = exp(x[2])

            if precip:
                jout[2, 0] = 1
                jout[2, 1] = 1
                jout[2, 2] = 0
            else:
                jout[2, 0] = 0
                jout[2, 1] = 0
                jout[2, 2] = 1

            return jout

        return NeqSys(3, 3, fun, jac,
                      pre_processors=[pre_processor],
                      post_processors=[post_processor])

    cneqsys = ConditionalNeqSys([
        (lambda x, p: x[0]*x[1] > p[3],
         lambda x, p: x[2] > exp(small))
    ], factory)

    _check_NaCl(cneqsys, [None], 4, method='lm')


def test_version():
    from pyneqsys import __version__
    assert int(__version__.split('.')[0]) >= 0


def test_solve_series():
    neqsys = NeqSys(1, 1, lambda x, p: [x[0]-p[0]])
    xout, sols = neqsys.solve_series('scipy', [0], [0], [0, 1, 2, 3], 0)
    assert np.allclose(xout[:, 0], [0, 1, 2, 3])
