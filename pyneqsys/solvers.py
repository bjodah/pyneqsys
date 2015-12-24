# -*- coding: utf-8 -*-
"""
Currently this module contains a few solvers for demonstration purposes and
currently it is not in the scope of pyneqsys to provide "production" solvers,
but instead be a package for using well established solvers for solving systems
of equations symbolically.

Nevertheless, the solvers provided here may provide a good starting point for
writing new types of solvers for systems of non-linear equations (with an
emphasis on a concise API rather than optimal performance).

Note that all classes and functions in this module (``pyneqsys.solvers``) are
provisional, i.e. they may be renamed, change behavior and or signature
without any prior notice.
"""
from __future__ import (absolute_import, division, print_function)

import numpy as np

def rms(x):
    return np.sqrt(np.mean(np.asarray(x)**2))


class SolverBase:

    def alloc(self):
        self.history_x = []
        self.history_dx = []
        self.history_f = []
        self.history_rms_f = []

    def step(self, x):
        pass

    def cb_factory(self):
        def cb(intern_x0, tol=1e-8, maxiter=100):
            cur_x = np.array(intern_x0)
            iter_idx = 0
            success = False
            self.history_x.append(cur_x.copy())
            while iter_idx < maxiter:
                f = np.asarray(self.inst.f_callback(
                    cur_x, self.inst.internal_params))
                self.history_f.append(f)
                rms_f = rms(f)
                self.history_rms_f.append(rms_f)

                self.history_dx.append(self.step(cur_x, iter_idx, maxiter))
                cur_x += self.history_dx[-1]
                self.history_x.append(cur_x.copy())

                iter_idx += 1
                if rms_f < tol:
                    success = True
                    break
            return cur_x, {'success': success}
        self.alloc()
        return cb

    def _gd_step(self, x):
        J = self.inst.j_callback(x, self.inst.internal_params)
        return -J.dot(self.history_f[-1])

    def __call__(self, inst):
        self.inst = inst
        self.cb = self.cb_factory()
        return self.cb


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

class PolakRibiereConjugateGradientSolver(SolverBase):

    def __init__(self, reset_freq=10):
        self.reset_freq = reset_freq

    def alloc(self):
        self.history_a = []  # for curiosity
        self.history_Bn = []  # for curiosity
        self.history_sn = []
        super(PolakRibiereConjugateGradientSolver, self).alloc()


    def line_search(self, x, dx):
        from scipy.optimize import fminbound
        return fminbound(lambda a: rms(self.inst.f_callback(
            x+a*dx, self.inst.internal_params)), 0, 1)

    def step(self, x, iter_idx, maxiter):
        dx = self.history_dx
        sn = self.history_sn
        if iter_idx in (0, 1) or iter_idx % self.reset_freq == 0:
            dxn = self._gd_step(x)
            dxn *= self.line_search(x, dxn)
            sn.append(x*0)
        else:
            dx0 = dx[-1]
            dx1 = dx[-2]
            ddx01 = dx0 - dx1
            Bn = Bn_suggest = dx0.dot(ddx01)/dx1.dot(dx1)
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


class QuorumSolver:
    pass
