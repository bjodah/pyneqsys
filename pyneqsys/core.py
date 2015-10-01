# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function


class NeqSys(object):
    """
    Object representing nonlinear equation system.
    Provides unified interface to:

    - scipy.optimize.root

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
        self.get_f_callback = lambda: f
        self.get_j_callback = lambda: jac
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

    def solve_scipy(self, x0, atol=1e-8, rtol=1e-8, method=None, **kwargs):
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
        atol: float
            Absolute tolerance
        rtol: float
            Relative tolerance
        method: str (default: None)
            what method to use.

        Returns
        -------
        array of length self.nx
        """
        from scipy.optimize import root
        f = self.get_f_callback()
        j = self.get_j_callback()
        if method is None:
            if self.nf > self.nx:
                method = 'lm'
            elif self.nf == self.nx:
                method = 'hybr'
            else:
                raise ValueError('Underdetermined problem')
        if 'band' in kwargs:
            raise ValueError("Set 'band' at initialization instead.")
        if self.band is not None:
            kwargs['band'] = self.band
        out = root(f, x0, jac=j, method=method)
        return self.post_process(out)
