# -*- coding: utf-8 -*-
"""
Currently this module contains a few solvers for demonstration purposes and
it is not in the scope of pyneqsys to provide "production" solvers,
but instead be a package for using well established solvers for solving systems
of equations defined in a uniform (optionally symbolic) way.

Nevertheless, the solvers provided here may serve as a starting point for
writing new types of solvers for systems of non-linear equations (with an
emphasis on a concise API rather than optimal performance).

Do not rely on any of the classes and functions in this module
(``pyneqsys.solvers``) since they are all to be regarded as provisional,
i.e. they may be renamed, change behavior and/or signature without any
prior notice.
"""

from __future__ import (absolute_import, division, print_function)

import numpy as np


def rms(x):
    return np.sqrt(np.mean(np.asarray(x)**2))


class SolverBase(object):

    def alloc(self):
        self.nfev = 0
        self.njev = 0
        self.history_x = []
        self.history_dx = []
        self.history_f = []
        self.history_rms_f = []

    def step(self, x):
        pass

    def f(self, x):
        self.nfev += 1
        return self.inst.f_callback(x, self.inst.internal_params)

    def j(self, x):
        self.njev += 1
        return self.inst.j_callback(x, self.inst.internal_params)

    def cb_factory(self):
        def cb(intern_x0, steptol=1e-8, ftol=1e-12, maxiter=100):
            if isinstance(ftol, float):
                ftol = ftol * np.ones_like(intern_x0)
            self.steptol = steptol
            self.ftol = ftol
            cur_x = np.array(intern_x0)
            iter_idx = 0
            success = False
            self.history_x.append(cur_x.copy())
            while iter_idx < maxiter:
                f = np.asarray(self.f(cur_x))
                self.history_f.append(f)
                rms_f = rms(f)
                self.history_rms_f.append(rms_f)

                self.history_dx.append(self.step(cur_x, iter_idx, maxiter))
                cur_x += self.history_dx[-1]
                self.history_x.append(cur_x.copy())

                iter_idx += 1
                if np.all(np.abs(f) < ftol):
                    success = True
                    break
            return {'x': cur_x, 'success': success, 'nit': iter_idx,
                    'nfev': self.nfev, 'njev': self.njev}
        self.alloc()
        return cb

    def __call__(self, inst):
        self.inst = inst
        self.cb = self.cb_factory()
        return self.cb

    def _gd_step(self, x):
        self.cur_j = self.j(x)
        return -self.cur_j.dot(self.history_f[-1])

    def line_search(self, x, dx, mxiter=10, alpha=1e-4):
        # Goldstein-Armijo linesearch (backtracking)
        idx = 0
        lmb = 1.0
        while idx < mxiter:
            f = self.f(x + lmb*dx)
            rms_f = rms(f)
            rms_cmp = rms(self.history_f[-1] + alpha*self.cur_j.dot(lmb*dx))
            if rms_f <= rms_cmp:
                return lmb*dx
            lmb /= 2
            idx += 1
        return lmb*dx


class GradientDescentSolver(SolverBase):
    """ Example of a custom solver

    Parameters
    ----------
    instance: NeqSys instance (passed by NeqSys.solve)

    Attributes
    ----------
    damping: callback
        with signature f(iter_idx, iter_max) -> float
        default: lambda idx, mx: exp(-idx/mx)/2

    Notes
    -----
    Note that this is an inefficient solver for demonstration purposes.
    """

    def damping(self, iter_idx, mx_iter):
        return 1.0

    def step(self, x, iter_idx, maxiter):
        return self.damping(iter_idx, maxiter) * self._gd_step(x)


class LineSearchingGradientDescentSolver(SolverBase):

    def step(self, x, iter_idx, maxiter):
        return self.line_search(x, self._gd_step(x))


class PolakRibiereConjugateGradientSolver(SolverBase):

    def __init__(self, reset_freq=10):
        self.reset_freq = reset_freq

    def alloc(self):
        self.history_a = []  # for curiosity
        self.history_Bn = []  # for curiosity
        self.history_sn = []
        super(PolakRibiereConjugateGradientSolver, self).alloc()

    def step(self, x, iter_idx, maxiter):
        dx = self.history_dx
        sn = self.history_sn
        if iter_idx in (0, 1) or iter_idx % self.reset_freq == 0:
            dxn = self.line_search(x, self._gd_step(x))
            sn.append(x*0)
        else:
            dx0 = dx[-1]
            dx1 = dx[-2]
            ddx01 = dx0 - dx1
            Bn = dx0.dot(ddx01)/dx1.dot(dx1)
            # print(Bn_suggest)
            # Bn = max(0, Bn_suggest)
            self.history_Bn.append(Bn)  # for curiosity
            sn.append(dx[-1] + Bn*sn[-1])
            a = self.line_search(x, sn[-1])
            self.history_a.append(a)  # for curiosity
            dxn = a*sn[-1]
        return dxn


class DampedGradientDescentSolver(GradientDescentSolver):

    def __init__(self, base_damp=.5, exp_damp=.5):
        self.base_damp = base_damp
        self.exp_damp = exp_damp

    def damping(self, iter_idx, mx_iter):
        import math
        return self.base_damp*math.exp(-iter_idx/mx_iter * self.exp_damp)


class AutoDampedGradientDescentSolver(GradientDescentSolver):

    def __init__(self, tgt_oscill=.1, start_damp=.1, nhistory=4, tgt_pow=.3):
        self.tgt_oscill = tgt_oscill
        self.cur_damp = start_damp
        self.nhistory = nhistory
        self.tgt_pow = tgt_pow
        self.history_damping = []

    def damping(self, iter_idx, mx_iter):
        if iter_idx >= self.nhistory:
            hist = self.history_rms_f[-self.nhistory:]
            avg = np.mean(hist)
            even = hist[::2]
            odd = hist[1::2]
            signed_metric = sum(even-avg) - sum(odd-avg)
            oscillatory_metric = abs(signed_metric)/(avg * self.nhistory)
            self.cur_damp *= (self.tgt_oscill/oscillatory_metric)**self.tgt_pow
        self.history_damping.append(self.cur_damp)
        return self.cur_damp


class QuorumSolver(object):
    pass
