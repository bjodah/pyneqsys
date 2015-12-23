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


class SolverBase:

    def step(self, x):
        pass

    def cb_factory(self):
        def cb(intern_x0, tol=1e-8, maxiter=100):
            cur_x = np.array(intern_x0)
            iter_idx = 0
            success = False
            cb.history_x.append(cur_x.copy())
            while iter_idx < maxiter:
                f = np.asarray(self.inst.f_callback(
                    cur_x, self.inst.internal_params))
                cb.history_f.append(f)
                rms_f = np.sqrt(np.mean(f**2))
                cb.history_rms_f.append(rms_f)

                cb.history_dx.append(self.step(cur_x, iter_idx, maxiter))
                cur_x += cb.history_dx[-1]
                cb.history_x.append(cur_x.copy())

                iter_idx += 1
                if rms_f < tol:
                    success = True
                    break
            return cur_x, {'success': success}
        cb.history_x = []
        cb.history_dx = []
        cb.history_f = []
        cb.history_rms_f = []
        return cb

    def _gd_step(self, x):
        J = self.inst.j_callback(x, self.inst.internal_params)
        return J.dot(self.history_f[-1])

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

    def __init__(self):
        self.history_sn = []

    def step(self, x, iter_idx, maxiter):
        dxn = self._gd_step(x)
        dx = self.cb.history_dx
        if iter_idx > 0:
            dx0 = dx[-1]
            dx1 = dx[-2]
            ddx01 = dx0 - dx1
            Bn = max(0, dx0.dot(ddx01)/dx1.dot(dx1))
            self.history_sn.append(dx + Bn*sn[-1])
            a = line_search(lambda a: self.inst.f_callback(
                x+a*self.history_sn[-1]))
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
            hist = self.cb.history_rms_f[-self.nhistory:]
            avg = np.mean(hist)
            even = hist[::2]
            odd = hist[1::2]
            signed_metrix = sum(even-avg) - sum(odd-avg)
            oscillatory_metric = abs(signed_metric)/(avg * self.nhistory)
            self.cur_damp *= (self.tgt_oscill/oscillatory_metric)**self.tgt_pow
        self.history_damping.append(self.cur_damp)
        return self.cur_damp


class QuorumSolver:
    pass
