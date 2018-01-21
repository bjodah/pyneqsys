# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import inspect
import math
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
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, fun):
            return fun


def _ensure_3args(func):
    if func is None:
        return None
    self_arg = 1 if inspect.ismethod(func) else 0
    if len(inspect.getargspec(func)[0]) == 3 + self_arg:
        return func
    if len(inspect.getargspec(func)[0]) == 2 + self_arg:
        return lambda x, params=(), backend=math: func(x, params)
    elif len(inspect.getargspec(func)[0]) == 1 + self_arg:
        return lambda x, params=(), backend=math: func(x)
    else:
        raise ValueError("Incorrect numer of arguments")


class _NeqSysBase(object):
    """ Baseclass for system of non-linear equations.

    This class contains shared logic used by its subclasses and is not meant to be used
    by end-users directly.
    """

    def __init__(self, names=None, param_names=None, x_by_name=None, par_by_name=None,
                 latex_names=None, latex_param_names=None):
        self.names = names or ()
        self.param_names = param_names or ()
        self.x_by_name = x_by_name
        self.par_by_name = par_by_name
        self.latex_names = latex_names or ()
        self.latex_param_names = latex_param_names or ()

    def _get_solver_cb(self, solver, attached_solver):
        if attached_solver is not None:
            if solver is not None:
                raise ValueError("solver must be None.")
            solver = attached_solver(self)
        if callable(solver):
            return solver
        if solver is None:
            solver = os.environ.get('PYNEQSYS_SOLVER', 'scipy')
        return getattr(self, '_solve_' + solver)

    def rms(self, x, params=()):
        """ Returns root mean square value of f(x, params) """
        internal_x, internal_params = self.pre_process(np.asarray(x),
                                                       np.asarray(params))
        if internal_params.ndim > 1:
            raise NotImplementedError("Parameters should be constant.")
        result = np.empty(internal_x.size//self.nx)
        for idx in range(internal_x.shape[0]):
            result[idx] = np.sqrt(np.mean(np.square(self.f_cb(
                internal_x[idx, :], internal_params))))
        return result

    def solve_series(self, x0, params, varied_data, varied_idx,
                     internal_x0=None, solver=None, propagate=True, **kwargs):
        """ Solve system for a set of parameters in which one is varied

        Parameters
        ----------
        x0 : array_like
            Guess (subject to ``self.post_processors``)
        params : array_like
            Parameter values
        vaired_data : array_like
            Numerical values of the varied parameter.
        varied_idx : int or str
            Index of the varied parameter (indexing starts at 0).
            If ``self.par_by_name`` this should be the name (str) of the varied
            parameter.
        internal_x0 : array_like (default: None)
            Guess (*not* subject to ``self.post_processors``).
            Overrides ``x0`` when given.
        solver : str or callback
            See :meth:`solve`.
        propagate : bool (default: True)
            Use last successful solution as ``x0`` in consecutive solves.
        \\*\\*kwargs :
            Keyword arguments pass along to :meth:`solve`.

        Returns
        -------
        xout : array
            Of shape ``(varied_data.size, x0.size)``.
        info_dicts : list of dictionaries
             Dictionaries each containing keys such as containing 'success', 'nfev', 'njev' etc.

        """
        if self.x_by_name and isinstance(x0, dict):
            x0 = [x0[k] for k in self.names]
        if self.par_by_name:
            if isinstance(params, dict):
                params = [params[k] for k in self.param_names]
            if isinstance(varied_idx, str):
                varied_idx = self.param_names.index(varied_idx)

        new_params = np.atleast_1d(np.array(params, dtype=np.float64))
        xout = np.empty((len(varied_data), len(x0)))
        self.internal_xout = np.empty_like(xout)
        self.internal_params_out = np.empty((len(varied_data),
                                             len(new_params)))
        info_dicts = []
        new_x0 = np.array(x0, dtype=np.float64)  # copy
        conds = kwargs.get('initial_conditions', None)  # see ConditionalNeqSys
        for idx, value in enumerate(varied_data):
            try:
                new_params[varied_idx] = value
            except TypeError:
                new_params = value  # e.g. type(new_params) == int
            if conds is not None:
                kwargs['initial_conditions'] = conds
            x, info_dict = self.solve(new_x0, new_params, internal_x0, solver,
                                      **kwargs)
            if propagate:
                if info_dict['success']:
                    try:
                        # See ChainedNeqSys.solve
                        new_x0 = info_dict['x_vecs'][0]
                        internal_x0 = info_dict['internal_x_vecs'][0]
                        conds = info_dict['intermediate_info'][0].get(
                            'conditions', None)
                    except:
                        new_x0 = x
                        internal_x0 = None
                        conds = info_dict.get('conditions', None)
            xout[idx, :] = x
            self.internal_xout[idx, :] = self.internal_x
            self.internal_params_out[idx, :] = self.internal_params
            info_dicts.append(info_dict)
        return xout, info_dicts

    def plot_series(self, xres, varied_data, varied_idx, **kwargs):
        """ Plots the results from :meth:`solve_series`.

        Parameters
        ----------
        xres : array
            Of shape ``(varied_data.size, self.nx)``.
        varied_data : array
            See :meth:`solve_series`.
        varied_idx : int or str
            See :meth:`solve_series`.
        \\*\\*kwargs :
            Keyword arguments passed to :func:`pyneqsys.plotting.plot_series`.

        """
        for attr in 'names latex_names'.split():
            if kwargs.get(attr, None) is None:
                kwargs[attr] = getattr(self, attr)
        ax = plot_series(xres, varied_data, **kwargs)
        if self.par_by_name and isinstance(varied_idx, str):
            varied_idx = self.param_names.index(varied_idx)
        if self.latex_param_names:
            ax.set_xlabel('$%s$' % self.latex_param_names[varied_idx])
        elif self.param_names:
            ax.set_xlabel(self.param_names[varied_idx])
        return ax

    def plot_series_residuals(self, xres, varied_data, varied_idx, params, **kwargs):
        """ Analogous to :meth:`plot_series` but will plot residuals. """
        nf = len(self.f_cb(*self.pre_process(xres[0], params)))
        xerr = np.empty((xres.shape[0], nf))
        new_params = np.array(params)

        for idx, row in enumerate(xres):
            new_params[varied_idx] = varied_data[idx]
            xerr[idx, :] = self.f_cb(*self.pre_process(row, params))
        return self.plot_series(xerr, varied_data, varied_idx, **kwargs)

    def plot_series_residuals_internal(self, varied_data, varied_idx, **kwargs):
        """ Analogous to :meth:`plot_series` but for internal residuals from last run. """
        nf = len(self.f_cb(*self.pre_process(
            self.internal_xout[0], self.internal_params_out[0])))
        xerr = np.empty((self.internal_xout.shape[0], nf))
        for idx, (res, params) in enumerate(zip(self.internal_xout, self.internal_params_out)):
            xerr[idx, :] = self.f_cb(res, params)
        return self.plot_series(xerr, varied_data, varied_idx, **kwargs)

    def solve_and_plot_series(self, x0, params, varied_data, varied_idx, solver=None, plot_kwargs=None,
                              plot_residuals_kwargs=None, **kwargs):
        """ Solve and plot for a series of a varied parameter.

        Convenience method, see :meth:`solve_series`, :meth:`plot_series` &
        :meth:`plot_series_residuals_internal` for more information.
        """
        sol, nfo = self.solve_series(
            x0, params, varied_data, varied_idx, solver=solver, **kwargs)
        ax_sol = self.plot_series(sol, varied_data, varied_idx, info=nfo,
                                  **(plot_kwargs or {}))

        extra = dict(ax_sol=ax_sol, info=nfo)
        if plot_residuals_kwargs:
            extra['ax_resid'] = self.plot_series_residuals_internal(
                varied_data, varied_idx, info=nfo,
                **(plot_residuals_kwargs or {})
            )
        return sol, extra


class NeqSys(_NeqSysBase):
    """Represents a system of non-linear equations.

    This class provides a unified interface to:

    - scipy.optimize.root
    - NLEQ2
    - KINSOL
    - mpmath
    - levmar

    Parameters
    ----------
    nf : int
        Number of functions.
    nx : int
        Number of independent variables.
    f : callback
        Function to solve for. Signature ``f(x) -> y`` where ``len(x) == nx``
        and ``len(y) == nf``.
    jac : callback or None (default)
        Jacobian matrix (dfdy).
    band : tuple (default: None)
        Number of sub- and super-diagonals in jacobian.
    names : iterable of str (default: None)
        Names of variables, used for plotting and for referencing by name.
    param_names : iterable of strings (default: None)
        Names of the parameters, used for referencing parameters by name.
    x_by_name : bool, default: ``False``
        Will values for *x* be referred to by name (in dictionaries)
        instead of by index (in arrays)?
    par_by_name : bool, default: ``False``
        Will values for parameters be referred to by name (in dictionaries)
        instead of by index (in arrays)?
    latex_names : iterable of str, optional
        Names of variables in LaTeX format.
    latex_param_names : iterable of str, optional
        Names of parameters in LaTeX format.
    pre_processors : iterable of callables (optional)
        (Forward) transformation of user-input to :py:meth:`solve`
        signature: ``f(x1[:], params1[:]) -> x2[:], params2[:]``.
        Insert at beginning.
    post_processors : iterable of callables (optional)
        (Backward) transformation of result from :py:meth:`solve`
        signature: ``f(x2[:], params2[:]) -> x1[:], params1[:]``.
        Insert at end.
    internal_x0_cb : callback (optional)
        callback with signature ``f(x[:], p[:]) -> x0[:]``
        if not specified, ``x`` from ``self.pre_processors`` will be used.

    Examples
    --------
    >>> neqsys = NeqSys(2, 2, lambda x, p: [(x[0] - x[1])**p[0]/2 + x[0] - 1,
    ...                                     (x[1] - x[0])**p[0]/2 + x[1]])
    >>> x, sol = neqsys.solve([1, 0], [3])
    >>> assert sol['success']
    >>> print(x)
    [ 0.8411639  0.1588361]

    See Also
    --------
    pyneqsys.symbolic.SymbolicSys : use a CAS (SymPy by default) to derive
                                    the jacobian.
    """

    def __init__(self, nf, nx=None, f=None, jac=None, band=None, pre_processors=None,
                 post_processors=None, internal_x0_cb=None, **kwargs):
        super(NeqSys, self).__init__(**kwargs)
        if nx is None:
            nx = len(self.names)
        if f is None:
            raise ValueError("A callback for f must be provided")
        if nf < nx:
            raise ValueError("Under-determined system")
        self.nf, self.nx = nf, nx
        self.f_cb = _ensure_3args(f)
        self.j_cb = _ensure_3args(jac)
        self.band = band
        self.pre_processors = pre_processors or []
        self.post_processors = post_processors or []
        self.internal_x0_cb = internal_x0_cb

    def pre_process(self, x0, params=()):
        """ Used internally for transformation of variables. """
        # Should be used by all methods matching "solve_*"
        if self.x_by_name and isinstance(x0, dict):
            x0 = [x0[k] for k in self.names]
        if self.par_by_name and isinstance(params, dict):
            params = [params[k] for k in self.param_names]
        for pre_processor in self.pre_processors:
            x0, params = pre_processor(x0, params)
        return x0, np.atleast_1d(params)

    def post_process(self, xout, params_out):
        """ Used internally for transformation of variables. """
        # Should be used by all methods matching "solve_*"
        for post_processor in self.post_processors:
            xout, params_out = post_processor(xout, params_out)
        return xout, params_out

    def solve(self, x0, params=(), internal_x0=None, solver=None, attached_solver=None, **kwargs):
        """ Solve with user specified ``solver`` choice.

        Parameters
        ----------
        x0: 1D array of floats
            Guess (subject to ``self.post_processors``)
        params: 1D array_like of floats
            Parameters (subject to ``self.post_processors``)
        internal_x0: 1D array of floats
            When given it overrides (processed) ``x0``. ``internal_x0`` is not
            subject to ``self.post_processors``.
        solver: str or callable or None or iterable of such
            if str: uses _solve_``solver``(\*args, \*\*kwargs).
            if ``None``: chooses from PYNEQSYS_SOLVER environment variable.
            if iterable: chain solving.
        attached_solver: callable factory
            Invokes: solver = attached_solver(self).

        Returns
        -------
        array:
            solution vector (post-processed by self.post_processors)
        dict:
            info dictionary containing 'success', 'nfev', 'njev' etc.

        Examples
        --------
        >>> neqsys = NeqSys(2, 2, lambda x, p: [
        ...     (x[0] - x[1])**p[0]/2 + x[0] - 1,
        ...     (x[1] - x[0])**p[0]/2 + x[1]
        ... ])
        >>> x, sol = neqsys.solve([1, 0], [3], solver=(None, 'mpmath'))
        >>> assert sol['success']
        >>> print(x)
        [0.841163901914009663684741869855]
        [0.158836098085990336315258130144]

        """
        if not isinstance(solver, (tuple, list)):
            solver = [solver]
        if not isinstance(attached_solver, (tuple, list)):
            attached_solver = [attached_solver] + [None]*(len(solver) - 1)
        _x0, self.internal_params = self.pre_process(x0, params)
        for solv, attached_solv in zip(solver, attached_solver):
            if internal_x0 is not None:
                _x0 = internal_x0
            elif self.internal_x0_cb is not None:
                _x0 = self.internal_x0_cb(x0, params)

            nfo = self._get_solver_cb(solv, attached_solv)(_x0, **kwargs)
            _x0 = nfo['x'].copy()
        self.internal_x = _x0
        x0 = self.post_process(self.internal_x, self.internal_params)[0]
        return x0, nfo

    def _solve_scipy(self, intern_x0, tol=1e-8, method=None, **kwargs):
        """ Uses ``scipy.optimize.root``

        See: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html

        Parameters
        ----------
        intern_x0: array_like
            initial guess
        tol: float
            Tolerance
        method: str
            What method to use. Defaults to ``'lm'`` if ``self.nf > self.nx`` otherwise ``'hybr'``.

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
        new_kwargs['args'] = self.internal_params

        return root(self.f_cb, intern_x0, jac=self.j_cb, method=method, tol=tol, **new_kwargs)

    def _solve_nleq2(self, intern_x0, tol=1e-8, method=None, **kwargs):
        from pynleq2 import solve

        def f_cb(x, ierr):
            f_cb.nfev += 1
            return self.f_cb(x, self.internal_params), ierr
        f_cb.nfev = 0

        def j_cb(x, ierr):
            j_cb.njev += 1
            return self.j_cb(x, self.internal_params), ierr
        j_cb.njev = 0

        x, ierr = solve(f_cb, j_cb, intern_x0, **kwargs)
        return {
            'x': x,
            'fun': np.asarray(f_cb(x, 0)),
            'success': ierr == 0,
            'nfev': f_cb.nfev,
            'njev': j_cb.njev,
            'ierr': ierr,
        }

    def _solve_kinsol(self, intern_x0, **kwargs):
        import pykinsol

        def _f(x, fout):
            res = self.f_cb(x, self.internal_params)
            fout[:] = res

        def _j(x, Jout, fx):
            res = self.j_cb(x, self.internal_params)
            Jout[:, :] = res[:, :]

        return pykinsol.solve(_f, _j, intern_x0, **kwargs)

    def _solve_mpmath(self, intern_x0, dps=30, tol=None,
                      maxsteps=None, **kwargs):
        import mpmath
        from mpmath.calculus.optimization import MDNewton
        mp = mpmath.mp
        mp.dps = dps

        def _mpf(val):
            try:
                return mp.mpf(val)
            except TypeError:  # e.g. mpmath chokes on numpy's int64
                return mp.mpf(float(val))
        intern_p = tuple(_mpf(_p) for _p in self.internal_params)
        maxsteps = maxsteps or MDNewton.maxsteps
        tol = tol or mp.eps * 1024

        def f_cb(*x):
            f_cb.nfev += 1
            return self.f_cb(x, intern_p)
        f_cb.nfev = 0

        if self.j_cb is not None:
            def j_cb(*x):
                j_cb.njev += 1
                return self.j_cb(x, intern_p)
            j_cb.njev = 0
            kwargs['J'] = j_cb
        intern_x0 = tuple(_mpf(_x) for _x in intern_x0)
        iters = MDNewton(mp, f_cb, intern_x0, norm=mp.norm, verbose=False, **kwargs)
        i = 0
        success = False
        for x, err in iters:
            i += 1
            lim = tol*max(mp.norm(x), 1)
            if err < lim:
                success = True
                break
            if i >= maxsteps:
                break
        result = {'x': x, 'success': success, 'nfev': f_cb.nfev, 'nit': i}
        if self.j_cb is not None:
            result['njev'] = j_cb.njev
        return result

    def _solve_ipopt(self, intern_x0, **kwargs):
        import warnings
        from ipopt import minimize_ipopt
        warnings.warn("ipopt interface has not yet undergone thorough testing.")

        def f_cb(x):
            f_cb.nfev += 1
            return np.sum(np.abs(self.f_cb(x, self.internal_params)))
        f_cb.nfev = 0

        if self.j_cb is not None:
            def j_cb(x):
                j_cb.njev += 1
                return self.j_cb(x, self.internal_params)
            j_cb.njev = 0
            kwargs['jac'] = j_cb

        return minimize_ipopt(f_cb, intern_x0, **kwargs)

    def _solve_levmar(self, intern_x0, tol=1e-8, **kwargs):
        import warnings
        import levmar

        if 'eps1' in kwargs or 'eps2' in kwargs or 'eps3' in kwargs:
            pass
        else:
            kwargs['eps1'] = kwargs['eps2'] = kwargs['eps3'] = tol

        def _f(*args):
            return np.asarray(self.f_cb(*args))

        def _j(*args):
            return np.asarray(self.j_cb(*args))

        _x0 = np.asarray(intern_x0)
        _y0 = np.zeros(self.nf)
        with warnings.catch_warnings(record=True) as wrns:
            warnings.simplefilter("always")
            p_opt, p_cov, info = levmar.levmar(_f, _x0, _y0, args=(self.internal_params,),
                                               jacf=_j, **kwargs)
        success = len(wrns) == 0 and np.all(np.abs(_f(p_opt, self.internal_params)) < tol)
        for w in wrns:
            raise w
        e2p0, (e2, infJTe, Dp2, mu_maxJTJii), nit, reason, nfev, njev, nlinsolv = info
        return {'x': p_opt, 'cov': p_cov, 'nfev': nfev, 'njev': njev, 'nit': nit,
                'message': reason, 'nlinsolv': nlinsolv, 'success': success}


class ConditionalNeqSys(_NeqSysBase):
    """ Collect multiple systems of non-linear equations with different
    conditionals.

    If a problem in a fixed number of variables is described by different
    systems of equations for different domains, then this class may be used
    to describe that set of systems.

    The user provides a set of conditions which governs what system of
    equations to apply. The set of conditions then represent a vector
    of booleans which is passed to a user provided factory function of
    NeqSys instances. The conditions may be asymmetrical (each condition
    consits of two callbacks, one for evaluating when the condition was
    previously ``False``, and one when it was previously ``True``. The motivation
    for this asymmetry is that a user may want to introduce a tolerance for
    numerical noise in the solution (and avoid possibly endless loops).

    If ``fastcache`` is available an LRU cache will be used for
    ``neqsys_factory``, it is therefore important that the factory function
    is idempotent.

    Parameters
    ----------
    condition_cb_pairs : list of (callback, callback) tuples
        Callbacks should have the signature: ``f(x, p) -> bool``.
    neqsys_factory : callback
        Should have the signature ``f(conds) -> NeqSys instance``,
        where conds is a list of bools.
    names : list of strings

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
    >>> x, sol = cneqsys.solve([0], [pi, 3])
    >>> assert sol['success']
    >>> print(x)
    [ 0.]
    >>> x, sol = cneqsys.solve([-1.4], [pi, 3])
    >>> assert sol['success']
    >>> print(x)
    [-1.]
    >>> x, sol = cneqsys.solve([2], [pi, 3])
    >>> assert sol['success']
    >>> print(x)
    [ 3.]
    >>> x, sol = cneqsys.solve([7], [pi, 3])
    >>> assert sol['success']
    >>> print(x)
    [ 3.]

    """

    def __init__(self, condition_cb_pairs, neqsys_factory, **kwargs):
        super(ConditionalNeqSys, self).__init__(**kwargs)
        self.condition_cb_pairs = condition_cb_pairs
        self.neqsys_factory = clru_cache(LRU_CACHE_SIZE)(neqsys_factory)

    def get_conds(self, x, params, prev_conds=None):
        if prev_conds is None:
            prev_conds = [False]*len(self.condition_cb_pairs)
        return tuple([bw(x, params) if prev else fw(x, params) for
                      prev, (fw, bw) in zip(prev_conds, self.condition_cb_pairs)])

    def solve(self, x0, params=(), internal_x0=None, solver=None,
              conditional_maxiter=20, initial_conditions=None, **kwargs):
        """ Solve the problem (systems of equations)

        Parameters
        ----------
        x0 : array
            Guess.
        params : array
            See :meth:`NeqSys.solve`.
        internal_x0 : array
            See :meth:`NeqSys.solve`.
        solver : str or callable or iterable of such.
            See :meth:`NeqSys.solve`.
        conditional_maxiter : int
            Maximum number of switches between conditions.
        initial_conditions : iterable of bools
            Corresponding conditions to ``x0``
        \\*\\*kwargs :
            Keyword arguments passed on to :meth:`NeqSys.solve`.

        """
        if initial_conditions is not None:
            conds = initial_conditions
        else:
            conds = self.get_conds(x0, params, initial_conditions)
        idx, nfev, njev = 0, 0, 0
        while idx < conditional_maxiter:
            neqsys = self.neqsys_factory(conds)
            x0, info = neqsys.solve(x0, params, internal_x0, solver, **kwargs)
            if idx == 0:
                internal_x0 = None
            nfev += info['nfev']
            njev += info.get('njev', 0)
            new_conds = self.get_conds(x0, params, conds)
            if new_conds == conds:
                break
            else:
                conds = new_conds
            idx += 1
        if idx == conditional_maxiter:
            raise Exception("Solving failed, conditional_maxiter reached")
        self.internal_x = info['x']
        self.internal_params = neqsys.internal_params
        result = {
            'x': info['x'],
            'success': info['success'],
            'conditions': conds,
            'nfev': nfev,
            'njev': njev,
        }
        if 'fun' in info:
            result['fun'] = info['fun']
        return x0, result

    def post_process(self, x, params, conds=None):
        if conds is None:
            conds = self.get_conds(x, params)
        return self.neqsys_factory(conds).post_process(x, params)

    post_process.__doc__ = NeqSys.post_process.__doc__

    def pre_process(self, x, params, conds=None):
        if conds is None:
            conds = self.get_conds(x, params)
        return self.neqsys_factory(conds).pre_process(x, params)

    pre_process.__doc__ = NeqSys.pre_process.__doc__

    def f_cb(self, x, params, conds=None):
        if conds is None:
            conds = self.get_conds(x, params)
        return self.neqsys_factory(conds).f_cb(x, params)


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
    >>> chained = ChainedNeqSys([neqsys_log, neqsys_lin])
    >>> x, info = chained.solve([1, 1], [4])
    >>> assert info['success']
    >>> print(x)
    [ 2.]
    >>> print(info['intermediate_info'][0]['nfev'],
    ...       info['intermediate_info'][1]['nfev'])  # doctest: +SKIP
    4 3

    """

    def __init__(self, neqsystems, **kwargs):
        super(ChainedNeqSys, self).__init__(**kwargs)
        self.neqsystems = neqsystems
        self.f_cb = self.neqsystems[0].f_cb

    def solve(self, x0, params=(), internal_x0=None, solver=None, **kwargs):
        x_vecs = []
        info_vec = []
        internal_x_vecs = []
        for idx, neqsys in enumerate(self.neqsystems):
            x0, info = neqsys.solve(x0, params, internal_x0, solver, **kwargs)
            if idx == 0:
                self.internal_x = info['x']
                self.internal_params = neqsys.internal_params
            internal_x0 = None  # only use for first iteration
            if 'conditions' in info:  # see ConditionalNeqSys.solve
                kwargs['initial_conditions'] = info['conditions']
            x_vecs.append(x0)
            internal_x_vecs.append(neqsys.internal_x)
            info_vec.append(info)
        info = {
            'x': self.internal_x,
            'success': info['success'],
            'nfev': sum([nfo['nfev'] for nfo in info_vec]),
            'njev': sum([nfo.get('njev', 0) for nfo in info_vec]),
        }
        if 'fun' in info:
            info['fun'] = info['fun']
        info['x_vecs'] = x_vecs
        info['intermediate_info'] = info_vec
        info['internal_x_vecs'] = internal_x_vecs
        return x0, info

    solve.__doc__ = NeqSys.solve.__doc__

    def post_process(self, x, params):
        return self.neqsystems[0].post_process(x, params)  # outermost

    post_process.__doc__ = NeqSys.post_process.__doc__

    def pre_process(self, x, params, conds=None):
        return self.neqsystems[0].pre_process(x, params)  # outermost

    pre_process.__doc__ = NeqSys.pre_process.__doc__
