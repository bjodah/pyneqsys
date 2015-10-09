# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from itertools import chain

from .core import NeqSys
from pyodesys.util import banded_jacobian, check_transforms


from pyodesys.symbolic import (
    _lambdify, _symarray, _num_transformer_factory
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
    lambdify: callback
        default: sympy.lambdify
    lambdify_unpack: bool (default: True)
        whether or not unpacking of args needed when calling lambdify callback
    Matrix: class
        default: sympy.Matrix
    \*\*kwargs:
        default kwargs to ``solve()``

    Notes
    -----
    Works for a moderate number of unknowns, sympy.lambdify has
    an upper limit on number of arguments.
    """

    def __init__(self, x, exprs, params=(), jac=True, band=None,
                 lambdify=None, lambdify_unpack=True, Matrix=None,
                 expand_params=False, **kwargs):
        self.x = x
        self.exprs = exprs
        self.params = params
        self._jac = jac
        self.band = band
        self.lambdify = lambdify or _lambdify
        self.lambdify_unpack = lambdify_unpack
        if Matrix is None:
            import sympy
            self.Matrix = sympy.ImmutableMatrix
        else:
            self.Matrix = Matrix
        self.expand_params = expand_params
        self.kwargs = kwargs

        self.f_callback = self.get_f_callback()
        self.j_callback = self.get_j_callback()

    @classmethod
    def from_callback(cls, cb, nx, nparams=-1, **kwargs):
        x = _symarray('x', nx)
        if nparams == -1:
            p = ()
            exprs = cb(x)
        else:
            p = _symarray('p', nparams)
            exprs = cb(x, p)
        return cls(x, exprs, p, **kwargs)

    @property
    def nx(self):
        return len(self.x)

    @property
    def nf(self):
        return len(self.exprs)

    def get_jac(self):
        if self._jac is True:
            if self.band is None:
                f = self.Matrix(1, self.nf, lambda _, q: self.exprs[q])
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
        cb = self.lambdify(list(chain(self.x, self.params)), self.exprs)

        def f(x, *args):
            new_args = chain(x, args[0] if self.expand_params else args)
            if self.lambdify_unpack:
                return cb(*new_args)
            else:
                return cb(new_args)
        return f

    def get_j_callback(self):
        cb = self.lambdify(list(chain(self.x, self.params)), self.get_jac())

        def j(x, *args):
            new_args = chain(x, args[0] if self.expand_params else args)
            if self.lambdify_unpack:
                return cb(*new_args)
            else:
                return cb(new_args)
        return j


class TransformedSys(SymbolicSys):

    def __init__(self, x, exprs, transf=None, **kwargs):
        if transf is not None:
            self.fw, self.bw = zip(*transf)
            check_transforms(self.fw, self.bw, x)
            exprs = [e.subs(zip(x, self.fw)) for e in exprs]
        else:
            self.fw, self.bw = None, None
        super(TransformedSys, self).__init__(x, exprs, **kwargs)

        self.fw_cb, self.bw_cb = _num_transformer_factory(self.fw, self.bw, x)
        self._pre_processor = lambda xarr: self.bw_cb(*xarr)
        self._post_processor = lambda xarr: self.fw_cb(*xarr)

    @classmethod
    def from_callback(cls, cb, nx, transf_cbs=None, **kwargs):
        x = _symarray('x', nx)
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


def linear_rref(A, b, Matrix=None):
    if Matrix is None:
        from sympy import Matrix
    aug = Matrix([list(row) + [v] for row, v in zip(A, b)])
    raug, pivot = aug.rref()
    nindep = len(pivot)
    return raug[:nindep, :-1], raug[:nindep, -1]


def linear_exprs(x, A, b, rref=False, Matrix=None):
    """
    returns Ax - b

    x: iterable of symbols
    A: matrix_like of numbers
        of shape (len(b), len(x))
    b: array_like of numbers
    Matrix: class
        When ``rref == Ture``: A matrix class which supports slicing,
        and methods ``dot`` and ``rref``. Defaults to sympy.Matrix
    rref: bool (default: False)
        calculate the reduced row echelon form of (A | -b)
    """
    if rref:
        rA, rb = linear_rref(A, b, Matrix)
        return [lhs - rhs for lhs, rhs in zip(rA.dot(x), rb)]
    else:
        return [sum([x0*x1 for x0, x1 in zip(row, x)]) - v
                for row, v in zip(A, b)]
