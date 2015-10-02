# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from .core import NeqSys
from pyodesys.util import banded_jacobian


from pyodesys.symbolic import (
    _lambdify, _Symbol, _symarray, _num_transformer_factory
)


class SymbolicSys(NeqSys):
    """
    Parameters
    ----------
    x: iterable of Symbols
    exprs: iterable of expressions for f
    jac: ImmutableMatrix or bool (default: True)
        If True:
            calculate jacobian from exprs
        If False:
            do not compute jacobian (numeric approximation)
        If ImmutableMatrix:
            user provided expressions for the jacobian
    band: tuple of two ints or None (default)
        number of lower and upper bands in jacobian.

    Notes
    -----
    Works for a moderate number of unknowns, sympy.lambdify has
    an upper limit on number of arguments.
    """

    def __init__(self, x, exprs, jac=True, band=None, lambdify=None,
                 Matrix=None):
        self.x = x
        self.exprs = exprs
        self._jac = jac
        self.band = band
        self.lambdify = lambdify or _lambdify
        if Matrix is None:
            import sympy
            self.Matrix = sympy.ImmutableMatrix
        else:
            self.Matrix = Matrix

    @classmethod
    def from_callback(cls, cb, nx, *args, **kwargs):
        x = _symarray('x', nx)
        exprs = cb(x)
        return cls(x, exprs, *args, **kwargs)

    @property
    def nx(self):
        return len(self.x)

    @property
    def nf(self):
        return len(self.exprs)

    def get_jac(self):
        if self._jac is True:
            if self.band is None:
                f = self.Matrix(1, self.nx, lambda _, q: self.exprs[q])
                return f.jacobian(self.x)
            else:
                # Banded
                return self.Matrix(banded_jacobian(
                    self.exprs, self.x, *self.band))
        elif self._jac is False:
            return False
        else:
            return self._jac

    def get_f_callback(self):
        cb = self.lambdify(self.x, self.exprs)
        return lambda x: cb(*x)

    def get_j_callback(self):
        cb = self.lambdify(self.x, self.get_jac())
        return lambda x: cb(*x)


class TransformedSys(SymbolicSys):

    def __init__(self, x, exprs, transf=None, **kwargs):
        if transf is not None:
            self.fw, self.bw = zip(*transf)
            exprs = [e.subs(self.fw) for e in exprs]
        else:
            self.fw, self.bw = None, None
        super(TransformedSys, self).__init__(x, exprs, **kwargs)

        self.fw_cb, self.bw_cb = _num_transformer_factory(self.fw, self.bw, x)
        self._pre_processor = self.fw_cb
        self._post_processor = self.back_transform_out

    @classmethod
    def from_callback(cls, cb, nx, transf_cbs=None, **kwargs):
        x = _Symbol('x', nx)
        exprs = cb(x)
        if transf_cbs is not None:
            try:
                transf = [(transf_cbs[idx][0](xi),
                           transf_cbs[idx][1](xi))
                          for idx, xi in enumerate(x)]
            except TypeError:
                transf = zip(map(transf_cbs[0], x), map(transf_cbs[1], x))
        else:
            transf = None
        return cls(x, exprs, transf, **kwargs)

    def back_transform_out(self, sol):
        sol.x[:] = self.bw_cb(sol.x)  # ugly
        return sol
