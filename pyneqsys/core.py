# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import inspect
import warnings

import numpy as np


def _ensure_2args(func):
    if len(inspect.getargspec(func)[0]) == 1:
        return lambda x, params: func(x)
    else:
        return func


def solve_series(solve, x0, params, var_data, var_idx, **kwargs):
    xout = np.empty((len(varied_data), len(x0)))
    sols = []
    new_x0 = np.array(x0, dtype=np.float64)
    new_params = np.atleast_1d(np.array(params, dtype=np.float64))
    for idx, value in enumerate(varied_data):
        try:
            new_params[idx_varied] = value
        except TypeError:
            new_params = value  # e.g. type(new_params) == int
        x, sol = solve(new_x0, new_params, **kwargs)
        if sol.success:
            new_x0 = x
        xout[idx, :] = x
        sols.append(sol)
    return xout, sols


class NeqSys(object):
    """
    Object representing nonlinear equation system.
    Provides unified interface to:

    - scipy.optimize.root
    - nleq2

    TODO: add more solvers, e.g. KINSOL.

    Parameters
    ----------
    nf: int
        number of functions
    nx: int
        number of parameters
    f: callback
        function to solve for signature f(x) where ``len(x) == nx``
        f should return an array_like of length ``nf``
    jac: callback or None (default)
        Jacobian matrix (dfdy). optional
    band: tuple (default: None)
        number of sub- and super-diagonals in jacobian.
    names: iterable of str (default: None)
        names of variables, used for plotting
    pre_processor: callback (array -> array)
        (forward) transformation of user-input to :py:meth:`solve`
    post_processor: callback (array -> array)
        (backward) transformation of result from :py:meth:`solve`
    """

    def __init__(self, nf, nx, f, jac=None, band=None, names=None,
                 pre_processor=None, post_processor=None):
        if nf < nx:
            raise ValueError("Under-determined system")
        self.nf, self.nx = nf, nx
        self.f_callback = _ensure_2args(f)
        self.j_callback = _ensure_2args(jac)
        self.band = band
        self.names = names
        self.pre_processor = pre_processor
        self.post_processor = post_processor


    def pre_process(self, x0):
        # Should be used by all methods matching "solve_*"
        if self.pre_processor is None:
            return x0
        else:
            return self.pre_processor(x0)

    def post_process(self, out):
        # Should be used by all methods matching "solve_*"
        if self.post_processor is None:
            return out
        else:
            return self.post_processor(out)

    def solve(self, solver, *args, **kwargs):
        """
        Solve with ``solver``. Convenience method.
        """
        return getattr(self, 'solve_'+solver)(*args, **kwargs)

    def solve_series(self, solver, x0, params, var_data, var_idx, **kwargs):
        return solve_series(getattr(self, 'solve_'+solver),
                            x0, params, var_data, var_idx, **kwargs)

    def solve_scipy(self, x0, params=None, tol=1e-8, method=None, **kwargs):
        """
        Use scipy.optimize.root
        see: http://docs.scipy.org/doc/scipy/reference/
                 generated/scipy.optimize.root.html

        Parameters
        ----------
        x0: array_like
            initial guess
        params: array_like (default: None)
            (Optional) parameters of type float64
        tol: float
            Tolerance
        method: str (default: None)
            what method to use.

        Returns
        -------
        array of length self.nx
        """
        from scipy.optimize import root
        if method is None:
            if self.nf > self.nx:
                method = 'lm'
            elif self.nf == self.nx:
                method = 'hybr'
            else:
                raise ValueError('Underdetermined problem')
        if 'band' in kwargs:
            raise ValueError("Set 'band' at initialization instead.")
        if 'args' in kwargs:
            raise ValueError("Set 'args' as params in initialization instead.")

        new_kwargs = kwargs.copy()
        if self.band is not None:
            warnings.warn("Band argument ignored (see SciPy docs)")
            new_kwargs['band'] = self.band
        if params is None:
            new_kwargs['args'] = []
        else:
            new_kwargs['args'] = np.atleast_1d(np.array(
                params, dtype=np.float64))

        sol = root(self.f_callback, self.pre_process(x0),
                   jac=self.j_callback, method=method, tol=tol,
                   **new_kwargs)

        return self.post_process(sol.x), sol

    def solve_nleq2(self, x0, params=None, tol=1e-8, method=None, **kwargs):
        """ Provisional, subject to unnotified API breaks """
        from pynleq2 import solve

        def f(x, ierr):
            return self.f_callback(x[:self.nx], x[self.nx:])
        x, ierr = solve(
            (lambda x, ierr: (self.f_callback(x, params), ierr)),
            (lambda x, ierr: (self.j_callback(x, params), ierr)),
            self.pre_process(x0),
            **kwargs
        )
        return self.post_process(x), ierr

    def plot_series(self, idx_varied, varied_data, xres, sols=None, plot=None,
                    plot_kwargs_cb=None, ls=('-', '--', ':', '-.'),
                    c=('k', 'r', 'g', 'b', 'c', 'm', 'y')):
        if plot is None:
            from matplotlib.pyplot import plot
        if plot_kwargs_cb is None:
            names = getattr(self, 'names', None)

            def plot_kwargs_cb(idx):
                kwargs = {'ls': ls[idx % len(ls)],
                          'c': c[idx % len(c)]}
                if names:
                    kwargs['label'] = names[idx]
                return kwargs
        else:
            plot_kwargs_cb = plot_kwargs_cb or (lambda idx: {})
        for idx in range(xres.shape[1]):
            plot(varied_data, xres[:, idx], **plot_kwargs_cb(idx))


class ConditionalNeqSys(object):
    """
    Parameters
    ----------
    conditions: list of (callback, callback) tuples
        callbacks should have the signature: f(x, p) -> bool
    neqsys_factory: callback
        should have the signature f(conds) -> NeqSys instance
        where conds is a list of bools
    """

    def __init__(self, conditions, neqsys_factory):
        self.conditions = conditions
        self.neqsys_factory = neqsys_factory

    def solve(self, solver, x0, params, conditional_maxiter=15, **kwargs):
        conds = [fw(x0, params) for fw, bw in self.conditions]
        idx = 0
        while idx < conditional_maxiter:
            print(conds, x0)
            neqsys = self.neqsys_factory(conds)
            x0, sol = neqsys.solve(solver, x0, params, **kwargs)
            new_conds = [not bw(x0, params) if prev else fw(x0, params)
                         for prev, (fw, bw) in zip(conds, self.conditions)]
            if new_conds == conds:
                break
            else:
                conds = new_conds
            idx += 1
        if idx == conditional_maxiter:
            raise Exception("Solving failed, conditional_maxiter reached")
        print(conds, x0)
        return x0, sol

    def solve_series(self, solver, x0, params, var_data, var_idx, **kwargs):
        return solve_series(lambda x, p, **kw: self.solve(solver, x, p, **kw),
                            x0, params, var_data, var_idx, **kwargs)
