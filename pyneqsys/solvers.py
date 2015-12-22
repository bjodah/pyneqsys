# -*- coding: utf-8 -*-
"""
Currently this module contains a few solvers for demonstration purposes and
currently it is not in the scope of pyneqsys to provide "production" solvers,
but instead be a package for using well established solvers for solving systems
of equations symbolically.

Nevertheless, the solvers provided here may provide a good starting point for
writing new types of solvers for systems of non-linear equations (with an
emphasis on a concise API rather than optimal performance).
"""
from __future__ import (absolute_import, division, print_function)

import numpy as np

class GradientDescentSolver:
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

    def damping(self, iter_idx, mx_iter, cb):
        return 1.0

    def __call__(self, inst):
        def cb(intern_x0, tol=1e-8, maxiter=100):
            cur_x = np.array(intern_x0)
            iter_idx = 0
            success = False
            cb.history_x.append(cur_x.copy())
            while iter_idx < maxiter:
                f = np.asarray(inst.f_callback(cur_x, inst.internal_params))
                cb.history_f.append(f)
                rms_f = np.sqrt(np.mean(f**2))
                cb.history_rms_f.append(rms_f)
                J = inst.j_callback(cur_x, inst.internal_params)
                cur_x -= self.damping(iter_idx, maxiter, cb) * J.dot(f)
                cb.history_x.append(cur_x.copy())
                iter_idx += 1
                if rms_f < tol:
                    success = True
                    break
            return cur_x, {'success': success}
        cb.history_x = []
        cb.history_f = []
        cb.history_rms_f = []
        return cb


class DampedGradientDescentSolver(GradientDescentSolver):
    def __init__(self, base_damp=.5, exp_damp=.5):
        self.base_damp = base_damp
        self.exp_damp = exp_damp

    def damping(self, iter_idx, mx_iter, cb):
        import math
        return self.base_damp*math.exp(-iter_idx/mx_iter * self.exp_damp)


class AutoDampedGradientDescentSolver(GradientDescentSolver):

    def __init__(self, target_oscill=.1, start_damp=.1, nhistory=4, tgt_pow=.3):
        self.target_oscill = target_oscill
        self.cur_damp = start_damp
        self.nhistory = nhistory
        self.tgt_pow = tgt_pow
        self.history_damping = []

    def damping(self, iter_idx, mx_iter, cb):
        if iter_idx >= self.nhistory:
            hist = cb.history_rms_f[-self.nhistory:]
            avg = np.mean(hist)
            even = hist[::2]
            odd = hist[1::2]
            oscillatory_metric = abs(sum(even-avg) - sum(odd-avg))/(avg * self.nhistory)
            self.cur_damp *= (self.target_oscill / oscillatory_metric)**self.tgt_pow
        self.history_damping.append(self.cur_damp)
        return self.cur_damp
