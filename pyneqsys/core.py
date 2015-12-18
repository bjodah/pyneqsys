# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import inspect
import os
import warnings

import numpy as np

from .plotting import plot_series

LRU_CACHE_SIZE = os.environ.get('PYNEQSYS_LRU_CACHE_SIZE', 256)

try:
    from fastcache import clru_cache
except ImportError:
    warnings.warn("Could not import 'fastcache' (look in PyPI), "
                  "solving of ConditionalNeqSys may be slower.")

    class clru_cache:
        def __init__(*args, **kwargs):
            pass

        def __call__(self, fun):
            return fun


def _ensure_2args(func):
    if func is None:
        return None

    if len(inspect.getargspec(func)[0]) == 1:
        return lambda x, params: func(x)
    else:
        return func


class _NeqSysBase(object):

    def _get_solver_cb(self, solver):
        if callable(solver):
            return solver
        if solver is None:
            solver = os.environ.get('PYNEQSYS_SOLVER', 'scipy')
        return getattr(self, '_solve_' + solver)

    def solve_series(self, solver, x0, params, varied_data, varied_idx,
                     internal_x0=None, **kwargs):
        new_params = np.atleast_1d(np.array(params, dtype=np.float64))
        xout = np.empty((len(varied_data), len(x0)))
        self.internal_xout = np.empty_like(xout)
        self.internal_params_out = np.empty((len(varied_data),
                                             len(new_params)))
        sols = []
        new_x0 = np.array(x0, dtype=np.float64)
        for idx, value in enumerate(varied_data):
            try:
                new_params[varied_idx] = value
            except TypeError:
                new_params = value  # e.g. type(new_params) == int
            x, sol = self.solve(solver, new_x0, new_params, internal_x0,
                                **kwargs)
            if sol['success']:
                try:
                    new_x0 = sol['sol_vecs'][0]  # See ChainedNeqSys.solve
                    internal_x0 = sol['internal_x_vecs'][0]
                except:
                    new_x0 = x
                    internal_x0 = None
            xout[idx, :] = x
            self.internal_xout[idx, :] = self.internal_x
            self.internal_params_out[idx, :] = self.internal_params
            sols.append(sol)
        return xout, sols

    def plot_series(self, *args, **kwargs):
        if kwargs.get('labels') is None:
            kwargs['labels'] = getattr(self, 'names', None)
        plot_series(*args, **kwargs)

    def plot_series_residuals(self, xres, varied_data, varied_idx, params,
                              **kwargs):
        xerr = np.empty((xres.shape[0], self.nf))
        new_params = np.array(params)
        for idx, row in enumerate(xres):
            new_params[varied_idx] = varied_data[idx]
            xerr[idx, :] = self.f_callback(*self.pre_process(row, params))
        self.plot_series(xerr, varied_data, labels=False, **kwargs)

    def plot_series_residuals_internal(self, varied_data, varied_idx,
                                       **kwargs):
        xerr = np.empty((self.internal_xout.shape[0], self.nf))
        for idx, (res, params) in enumerate(zip(self.internal_xout,
                                                self.internal_params_out)):
            xerr[idx, :] = self.f_callback(res, params)
        self.plot_series(xerr, varied_data, labels=False, **kwargs)

    def solve_and_plot_series(self, solver, x0, params, varied_data,
                              varied_idx,
                              plot_series_ax=None,
                              plot_series_residuals_ax=None,
                              plot_series_kwargs=None,
                              plot_series_residuals_kwargs=None,
                              **kwargs):
        """ Solve and plot for a series of a varied parameter """
        xres, sols = self.solve_series(solver, x0, params, varied_data,
                                       varied_idx, **kwargs)
        self.plot_series(xres, varied_data, sols=sols, ax=plot_series_ax,
                         **(plot_series_kwargs or {}))
        if plot_series_residuals_ax is not None:
            self.plot_series_residuals_internal(
                varied_data, varied_idx, sols=sols,
                ax=plot_series_residuals_ax,
                **(plot_series_residuals_kwargs or {})
            )
        return xres, sols


class NeqSys(_NeqSysBase):
    """Represent a system of non-linear equations

    Object representing nonlinear equation system.
    Provides unified interface to:

    - scipy.optimize.root
    - nleq2

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
    pre_processors: iterable of callables (optional)
        (forward) transformation of user-input to :py:meth:`solve`
        signature: f(x1[:], params1[:]) -> x2[:], params2[:]
        insert at beginning
    post_processors: iterable of callables (optional)
        (backward) transformation of result from :py:meth:`solve`
        signature: f(x2[:], params2[:]) -> x1[:], params1[:]
        insert at end

    Examples
    --------
    >>> neqsys = NeqSys(2, 2, lambda x, p: [(x[0] - x[1])**p[0]/2 + x[0] - 1,
    ...                                     (x[1] - x[0])**p[0]/2 + x[1]])
    >>> x, sol = neqsys.solve('scipy', [1, 0], [3])
    >>> assert sol['success']
    >>> print(x)
    [ 0.8411639  0.1588361]

    See Also
    --------
    pyneqsys.symbolic.SymbolicSys : use a CAS (SymPy by default) to derive
                                    the jacobian.
    """

    def __init__(self, nf, nx, f, jac=None, band=None, names=None,
                 pre_processors=None, post_processors=None):
        if nf < nx:
            raise ValueError("Under-determined system")
        self.nf, self.nx = nf, nx
        self.f_callback = _ensure_2args(f)
        self.j_callback = _ensure_2args(jac)
        self.band = band
        self.names = names
        self.pre_processors = pre_processors or []
        self.post_processors = post_processors or []

    def pre_process(self, x0, params=()):
        """ Used internally for transformation of variables """
        # Should be used by all methods matching "solve_*"
        for pre_processor in self.pre_processors:
            x0, params = pre_processor(x0, params)
        return x0, params

    def post_process(self, xout, params_out):
        """ Used internally for transformation of variables """
        # Should be used by all methods matching "solve_*"
        for post_processor in self.post_processors:
            xout, params_out = post_processor(xout, params_out)
        return xout, params_out

    def solve(self, solver, x0, params=(), internal_x0=None, **kwargs):
        """
        Solve with ``solver``. Convenience method.

        Parameters
        ----------
        solver: str or None
            if str: uses _solve_``solver``(\*args, \*\*kwargs)
            if ``None``: chooses from PYNEQSYS_SOLVER environment variable
        x0: 1D array of floats
            Guess (subject to ``self.post_processors``)
        params: 1D array_like of floats (default: ())
            Parameters (subject to ``self.post_processors``)
        internal_x0: 1D array of floats (default: None)
            When given it overrides (processed) ``x0``. ``internal_x0`` is not
            subject to ``self.post_processors``.
        """
        intern_x0, self.internal_params = self.pre_process(x0, params)
        if internal_x0 is not None:
            intern_x0 = internal_x0
        self.internal_x, sol = self._get_solver_cb(solver)(intern_x0, **kwargs)
        return self.post_process(self.internal_x,
                                 self.internal_params)[:1] + (sol,)

    def _solve_scipy(self, intern_x0, tol=1e-8, method=None, **kwargs):
        """
        Use ``scipy.optimize.root``
        see: http://docs.scipy.org/doc/scipy/reference/\
generated/scipy.optimize.root.html

        Parameters
        ----------
        intern_x0: array_like
            initial guess
        tol: float
            Tolerance
        method: str (default: None)
            what method to use.

        Returns
        -------
        Length 2 tuple:
           - solution (array of length self.nx)
           - additional output from solver

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
        new_kwargs['args'] = np.atleast_1d(np.array(
            self.internal_params, dtype=np.float64))

        sol = root(self.f_callback, intern_x0,
                   jac=self.j_callback, method=method, tol=tol,
                   **new_kwargs)
        return sol.x, sol

    def _solve_nleq2(self, intern_x0, tol=1e-8, method=None, **kwargs):
        """ Provisional, subject to unnotified API breaks """
        from pynleq2 import solve

        def f(x, ierr):
            return self.f_callback(x[:self.nx], x[self.nx:])
        x, ierr = solve(
            (lambda x, ierr: (self.f_callback(x, self.internal_params), ierr)),
            (lambda x, ierr: (self.j_callback(x, self.internal_params), ierr)),
            intern_x0,
            **kwargs
        )
        self.internal_x = x
        return self.post_process(x, self.internal_params)[:1] + (
            {'ierr': ierr},)


class ConditionalNeqSys(_NeqSysBase):
    """ Collect multiple systems of non-linear equations with different
    conditionals.

    If a problem in a fixed number of variables is described by different
    systems of equations this class may be used to describe that set of
    systems.

    The user provides a set of conditions which governs what system of
    equations to apply. The set of conditions then represent a vector
    of booleans which is passed to a user provided NeqSys-factory.
    The conditions may be asymmetrical (each condition consits of two
    callbacks, one for evaluating when the condition was previously False,
    and one when it was previously False. The motivation for this asymmetry
    is that a user may want to introduce a tolerance for numerical noise in
    the solution (and avoid possibly infinite recursion).

    If ``fastcache`` is available an LRU cache will be used for
    ``neqsys_factory``, it is therefore important that the function is
    idempotent.

    Parameters
    ----------
    condition_cb_pairs: list of (callback, callback) tuples
        callbacks should have the signature: f(x, p) -> bool
    neqsys_factory: callback
        should have the signature f(conds) -> NeqSys instance
        where conds is a list of bools

    Examples
    --------
    >>> from math import sin, pi
    >>> f_a = lambda x, p: [sin(p[0]*x[0])]  # when x <= 0
    >>> f_b = lambda x, p: [x[0]*(p[1]-x[0])]  # when x >= 0
    >>> factory = lambda conds: NeqSys(1, 1, f_b) if conds[0] else NeqSys(
    ...     1, 1, f_a)
    >>> cneqsys = ConditionalNeqSys([(lambda x, p: x[0] > 0,  # no 0-switch
    ...                               lambda x, p: x[0] >= 0)],  # no 0-switch
    ...                             factory)
    >>> x, sol = cneqsys.solve('scipy', [0], [pi, 3])
    >>> assert sol['success']
    >>> print(x)
    [ 0.]
    >>> x, sol = cneqsys.solve('scipy', [-1.4], [pi, 3])
    >>> assert sol['success']
    >>> print(x)
    [-1.]
    >>> x, sol = cneqsys.solve('scipy', [2], [pi, 3])
    >>> assert sol['success']
    >>> print(x)
    [ 3.]
    >>> x, sol = cneqsys.solve('scipy', [7], [pi, 3])
    >>> assert sol['success']
    >>> print(x)
    [ 3.]

    """

    def __init__(self, condition_cb_pairs, neqsys_factory, names=None):
        self.condition_cb_pairs = condition_cb_pairs
        self.neqsys_factory = clru_cache(LRU_CACHE_SIZE)(neqsys_factory)
        self.names = names

    def get_conds(self, x, params, prev_conds=None):
        if prev_conds is None:
            prev_conds = [False]*len(self.condition_cb_pairs)
        return tuple([bw(x, params) if prev else fw(x, params)
                      for prev, (fw, bw) in zip(prev_conds, self.condition_cb_pairs)])

    def solve(self, solver, x0, params=(), internal_x0=None,
              conditional_maxiter=20, initial_conditions=None, **kwargs):
        """ Solve the problem (systems of equations) """
        conds = self.get_conds(x0, params, initial_conditions)  # DO-NOT-MERGE!
        if initial_conditions is not None:  # this is one alternative
            conds = initial_conditions      # (if I keep this: remove above)
        idx = 0
        while idx < conditional_maxiter:
            neqsys = self.neqsys_factory(conds)
            x0, sol = neqsys.solve(solver, x0, params, internal_x0, **kwargs)
            internal_x0 = None
            nconds = self.get_conds(x0, params, conds)
            if nconds == conds:
                break
            else:
                conds = nconds
            idx += 1
        if idx == conditional_maxiter:
            raise Exception("Solving failed, conditional_maxiter reached")
        # print('conditional iter:', idx)#debugging
        self.internal_x = x0
        self.internal_params = params
        return x0, {'success': sol['success'], 'conditions': conds}

    def post_process(self, x, params, conds=None):
        if conds is None:
            conds = self.get_conds(x, params)
        return self.neqsys_factory(conds).post_process(x, params)

    def pre_process(self, x, params, conds=None):
        if conds is None:
            conds = self.get_conds(x, params)
        return self.neqsys_factory(conds).pre_process(x, params)


class ChainedNeqSys(_NeqSysBase):
    """ Chain multiple formulations of non-linear systems for using
    the result of one as starting guess for the other

    Examples
    --------
    >>> neqsys_lin = NeqSys(1, 1, lambda x, p: [x[0]**2 - p[0]])
    >>> from math import log, exp
    >>> neqsys_log = NeqSys(1, 1, lambda x, p: [2*x[0] - log(p[0])],
    ...    pre_processors=[lambda x, p: ([log(x[0]+1e-60)], p)],
    ...    post_processors=[lambda x, p: ([exp(x[0])], p)])
    >>> chained = ChainedNeqSys([neqsys_log, neqsys_lin], save_sols=True)
    >>> x, sol = chained.solve('scipy', [1, 1], [4])
    >>> assert sol['success']
    >>> print(x)
    [ 2.]
    >>> print(chained.last_solve_sols[0].nfev,
    ...       chained.last_solve_sols[1].nfev)  # doctest: +SKIP
    4 3

    """

    def __init__(self, neqsystems, save_sols=False, names=None):
        self.neqsystems = neqsystems
        self.save_sols = save_sols
        self.names = names

    def solve(self, solver, x0, params=(), internal_x0=None, **kwargs):
        if self.save_sols:
            self.last_solve_sols = []
        # print('x0', x0)##DEBUG
        sol_vecs = []
        internal_x_vecs = []
        for idx, neqsys in enumerate(self.neqsystems):
            x0, sol = neqsys.solve(solver, x0, params, internal_x0, **kwargs)
            internal_x0 = None  # only use for first iteration
            if 'conditions' in sol:  # see ConditionalNeqSys.solve
                kwargs['initial_conditions'] = sol['conditions']
            sol_vecs.append(x0)
            internal_x_vecs.append(neqsys.internal_x)
            if self.save_sols:
                self.last_solve_sols.append(sol)
        self.internal_x = x0
        self.internal_params = params
        info_dict = {'success': sol['success']}
        info_dict['sol_vecs'] = sol_vecs
        info_dict['internal_x_vecs'] = internal_x_vecs
        return x0, info_dict

    @classmethod
    def from_callback(cls, NeqSys_vec, *args, **kwargs):
        return cls([NS.from_callback(*args, **kwargs) for NS in NeqSys_vec])

    def post_process(self, x, params):
        return self.neqsystems[0].post_process(x, params)  # outermost

    def pre_process(self, x, params, conds=None):
        return self.neqsystems[0].pre_process(x, params)  # outermost
