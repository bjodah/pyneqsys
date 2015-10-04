# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from itertools import chain
import warnings


class NeqSys(object):
    """
    Object representing nonlinear equation system.
    Provides unified interface to:

    - scipy.optimize.root

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
    """

    _pre_processor = None
    _post_processor = None

    def __init__(self, nf, nx, f, jac=None, band=None,
                 **kwargs):
        if nf < nx:
            raise ValueError("Under-determined system")
        self.nf, self.nx = nf, nx
        self.f_callback = f
        self.j_callback = jac
        self.band = band
        self.kwargs = kwargs  # default kwargs for integration

    def pre_process(self, x0):
        # Should be used by all methods matching "solve_*"
        if self._pre_processor is None:
            return x0
        else:
            return self._pre_processor(x0)

    def post_process(self, out):
        # Should be used by all methods matching "solve_*"
        if self._post_processor is None:
            return out
        else:
            return self._post_processor(out)

    def solve(self, solver, *args, **kwargs):
        """
        Solve with ``solver``. Convenience method.
        """
        return getattr(self, 'solve_'+solver)(*args, **kwargs)

    def solve_scipy(self, x0, params=None, tol=1e-8, method=None, **kwargs):
        """
        Use scipy.optimize.root
        see: http://docs.scipy.org/doc/scipy/reference/
                 generated/scipy.optimize.root.html

        Parameters
        ----------
        x0: array_like
            initial guess
        y0: array_like
            Initial values at xout[0] for the dependent variables.
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

        kwargs = dict(chain(self.kwargs.items(), kwargs.items()))
        if self.band is not None:
            warnings.warn("Band argument ignored (see SciPy docs)")
            kwargs['band'] = self.band
        if params is not None:
            kwargs['args'] = params

        sol = root(self.f_callback, self.pre_process(x0),
                   jac=self.j_callback, method=method, tol=tol,
                   **kwargs)

        return self.post_process(sol.x), sol
