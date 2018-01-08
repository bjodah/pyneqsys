# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import math
import pytest

import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys

try:
    import pynleq2  # noqa
except ImportError:
    HAVE_PYNLEQ2 = False
else:
    HAVE_PYNLEQ2 = True


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


def _test_fail(solver, **kwargs):
    def _f(x, p):
        return [p[0] + x[0]**2]

    def _j(x, p):
        return [[2*x[0]]]

    ns = NeqSys(1, 1, _f, jac=_j)
    x, res = ns.solve([1], [1], solver=solver, **kwargs)
    assert len(x) == 1
    assert abs(x[0]) < 1e-8
    assert not res['success']


def test_fail():
    _test_fail('scipy')


def test_neqsys_rms():
    ns = NeqSys(2, 2, f)
    x = [[1, 0], [2, 1], [3, 2], [7, 4], [5, 13]]
    p = [3]
    rms = ns.rms(x, p)
    ref = [np.sqrt(np.sum(np.square(f(x[i], p)))/2) for i in range(5)]
    assert np.allclose(rms, ref)


def _test_neqsys_params(solver, **kwargs):
    ns = NeqSys(2, 2, f, jac=j)
    x, sol = ns.solve([0, 0], [3], solver=solver, **kwargs)
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


@pytest.mark.skipif(not HAVE_PYNLEQ2, reason='pynleq2 not installed on system.')
def test_neqsys_params_nleq2():
    _test_neqsys_params('nleq2')


def _test_neqsys_no_params(solver, **kwargs):
    ns = NeqSys(2, 2, lambda x: f(x, [3]),
                jac=lambda x: j(x, [3]))
    x, sol = ns.solve([0, 0], solver=solver, **kwargs)
    assert abs(x[0] - 0.8411639) < 2e-7
    assert abs(x[1] - 0.1588361) < 2e-7


def test_neqsys_no_params_scipy():
    _test_neqsys_no_params('scipy')


@pytest.mark.skipif(not HAVE_PYNLEQ2, reason='pynleq2 not installed on system.')
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
    x, sol = cneqsys.solve([0], [pi, 3], solver='scipy')
    assert sol['success']
    assert abs(x[0]) < 1e-13
    x, sol = cneqsys.solve([-1.4], [pi, 3], solver='scipy')
    assert sol['success']
    assert abs(x[0] + 1) < 1e-13
    x, sol = cneqsys.solve([2], [pi, 3], solver='scipy')
    assert sol['success']
    assert abs(x[0] - 3) < 1e-13
    x, sol = cneqsys.solve([7], [pi, 3], solver='scipy')
    assert sol['success']
    assert abs(x[0] - 3) < 1e-13


def _check_NaCl(cneqsys, guesses, cases=-1, **kwargs):
    # Sodium chloride (NaCl) precipitation
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
    info_dicts = []
    for init, final in _init_final[:cases]:
        print(init)
        for guess in guesses:
            print(guess)
            if guess is None:
                guess = init
            x, info_dict = cneqsys.solve(guess, init + [4],
                                         solver='scipy', **kwargs)
            assert info_dict['success'] and np.allclose(x, final)
            info_dicts.append(info_dict)
    return info_dicts


def _factory_lin(conds):
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


def _factory_log(small):
    # This is equivalent to _factory_lin
    # but this time variable transformations
    # are performed

    def _inner_factory(conds):
        precip = conds[0]

        def pre_processor(x, p):
            return np.log(np.asarray(x) + math.exp(small)), p

        def post_processor(x, p):
            return np.exp(x), p

        def fun(x, p):
            f = [None]*3
            f[0] = math.exp(x[0]) + math.exp(x[2]) - p[0] - p[2]
            f[1] = math.exp(x[1]) + math.exp(x[2]) - p[1] - p[2]
            if precip:
                f[2] = x[0] + x[1] - math.log(p[3])
            else:
                f[2] = x[2] - small
            return f

        def jac(x, p):
            jout = np.empty((3, 3))

            jout[0, 0] = math.exp(x[0])
            jout[0, 1] = 0
            jout[0, 2] = math.exp(x[2])

            jout[1, 0] = 0
            jout[1, 1] = math.exp(x[1])
            jout[1, 2] = math.exp(x[2])

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

    return _inner_factory


def _get_cneqsys2():
    return ConditionalNeqSys([
        (lambda x, p: x[0]*x[1] > p[3],
         lambda x, p: x[2] > 0)
    ], _factory_lin)


def test_ConditionalNeqSys2():
    _check_NaCl(_get_cneqsys2(), [(1, 1, 1), (1, 1, 0), (2, 2, 0), (1, 1, 3)])


def _get_cneqsys3(small):
    return ConditionalNeqSys([
        (lambda x, p: x[0]*x[1] > p[3],
         lambda x, p: x[2] > math.exp(small))
    ], _factory_log(small))


def test_ConditionalNeqSys3():
    _check_NaCl(_get_cneqsys3(-60), [None], 4, method='lm')


def test_version():
    from pyneqsys import __version__
    assert int(__version__.split('.')[0]) >= 0


def test_solve_series():
    neqsys = NeqSys(1, 1, lambda x, p: [x[0]-p[0]])
    xout, sols = neqsys.solve_series([0], [0], [0, 1, 2, 3], 0, solver='scipy')
    assert np.allclose(xout[:, 0], [0, 1, 2, 3])


def test_ChainedNeqSys():
    neqsys_log = _get_cneqsys3(-60)
    neqsys_lin = _get_cneqsys2()
    chained = ChainedNeqSys([neqsys_log, neqsys_lin])
    info_dicts = _check_NaCl(chained, [None], 2, method='lm')
    for nfo in info_dicts:
        assert (nfo['intermediate_info'][0]['success'] and
                nfo['intermediate_info'][1]['success'])


_powell_ref = (0.0001477105829534399, 6.769995622556115071)


def _powell(x, params, backend=math):
    A, exp = params[0], backend.exp
    return A*x[0]*x[1] - 1, exp(-x[0]) + exp(-x[1]) - (1 + A**-1)


def _test_powell(sys_solver_pairs, x0=(1, 1), par=(1000.0,)):
    for sys, solver in sys_solver_pairs:
        x0, info = sys.solve(x0, par, solver=solver, tol=1e-12)
    assert info['success']
    x = sorted(x0)
    assert abs(_powell_ref[0] - x[0]) < 2e-11
    assert abs(_powell_ref[1] - x[1]) < 6e-10


def test_chained_solvers():
    powell_numpy = NeqSys(2, 2, _powell)
    powell_mpmath = NeqSys(2, 2, _powell)
    _test_powell([(powell_numpy, None), (powell_mpmath, 'mpmath')])


def test_x_by_name():
    powell_sys = NeqSys(2, f=_powell, names=['u', 'v'], x_by_name=True)
    _test_powell(zip([powell_sys]*2, [None, 'mpmath']), x0={'u': 1, 'v': 1})
