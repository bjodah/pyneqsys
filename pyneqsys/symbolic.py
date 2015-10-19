# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from itertools import chain
import numpy as np
import sympy as sp

from .core import NeqSys, _ensure_2args
from pyodesys.util import banded_jacobian, check_transforms


def _lambdify(*args, **kwargs):
    if 'modules' not in kwargs:
        kwargs['modules'] = [{'ImmutableMatrix': np.array}, 'numpy']
    return sp.lambdify(*args, **kwargs)


def _symarray(prefix, shape, Symbol=None):
    # see https://github.com/sympy/sympy/pull/9939
    # when released: return sp.symarray(key, n, real=True)
    arr = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        arr[index] = (Symbol or (lambda name: sp.Symbol(name, real=True)))(
            '%s_%s' % (prefix, '_'.join(map(str, index))))
    return arr


def _num_transformer_factory(fw, bw, dep, lambdify=None):
    lambdify = lambdify or _lambdify
    return lambdify(dep, fw), lambdify(dep, bw)


class SymbolicSys(NeqSys):
    """
    Parameters
    ----------
    x: iterable of Symbols
    exprs: iterable of expressions for f
    params: iterable of Symbols (optional)
        list of symbols appearing in exprs which are parameters
    jac: ImmutableMatrix or bool (default: True)
        If True:
            calculate jacobian from exprs
        If False:
            do not compute jacobian (numeric approximation)
        If ImmutableMatrix:
            user provided expressions for the jacobian
    lambdify: callback
        default: sympy.lambdify
    lambdify_unpack: bool (default: True)
        whether or not unpacking of args needed when calling lambdify callback
    Matrix: class
        default: sympy.Matrix
    \*\*kwargs:
        See :py:class:`NeqSys`

    Notes
    -----
    Works for a moderate number of unknowns, sympy.lambdify has
    an upper limit on number of arguments.
    """

    def __init__(self, x, exprs, params=(), jac=True, lambdify=None,
                 lambdify_unpack=True, Matrix=None, **kwargs):
        self.x = x
        self.exprs = exprs
        self.params = params
        self._jac = jac
        self.lambdify = lambdify or _lambdify
        self.lambdify_unpack = lambdify_unpack
        if Matrix is None:
            import sympy
            self.Matrix = sympy.ImmutableMatrix
        else:
            self.Matrix = Matrix
        self.nf, self.nx = len(exprs), len(x)  # needed by get_*_callback
        self.band = kwargs.get('band', None)  # needed by get_*_callback
        super(SymbolicSys, self).__init__(self.nf, self.nx,
                                          self.get_f_callback(),
                                          self.get_j_callback(),
                                          **kwargs)

    @classmethod
    def from_callback(cls, cb, nx, nparams=0, **kwargs):
        x = _symarray('x', nx)
        p = _symarray('p', nparams)
        if nparams == 0:
            cb = _ensure_2args(cb)
        exprs = cb(x, p)
        return cls(x, exprs, p, **kwargs)

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

        def f(x, params):
            new_args = list(chain(x, params))
            if self.lambdify_unpack:
                return cb(*new_args)
            else:
                return cb(new_args)
        return f

    def get_j_callback(self):
        cb = self.lambdify(list(chain(self.x, self.params)), self.get_jac())

        def j(x, params):
            new_args = chain(x, params)
            if self.lambdify_unpack:
                return cb(*new_args)
            else:
                return cb(new_args)
        return j


class TransformedSys(SymbolicSys):

    def __init__(self, x, exprs, transf=None, params=(), **kwargs):
        if transf is not None:
            self.fw, self.bw = zip(*transf)
            check_transforms(self.fw, self.bw, x)
            exprs = [e.subs(zip(x, self.fw)) for e in exprs]
        else:
            self.fw, self.bw = None, None
        super(TransformedSys, self).__init__(x, exprs, params, **kwargs)

        self.fw_cb, self.bw_cb = _num_transformer_factory(self.fw, self.bw, x)
        self._pre_processor = lambda xarr: self.bw_cb(*xarr)
        self._post_processor = lambda xarr: self.fw_cb(*xarr)

    @classmethod
    def from_callback(cls, cb, nx, nparams=0, exprs_transf=None,
                      transf_cbs=None, **kwargs):
        x = _symarray('x', nx)
        p = _symarray('p', nparams)
        if nparams == 0:
            cb = _ensure_2args(cb)
        exprs = cb(x, p)
        if exprs_transf is not None:
            exprs = [exprs_transf(expr) for expr in exprs]
        if transf_cbs is not None:
            try:
                transf = [(transf_cbs[idx][0](xi),
                           transf_cbs[idx][1](xi))
                          for idx, xi in enumerate(x)]
            except TypeError:
                transf = zip(map(transf_cbs[0], x), map(transf_cbs[1], x))
        else:
            transf = None
        return cls(x, exprs, transf, p, **kwargs)


def linear_rref(A, b, Matrix=None):
    if Matrix is None:
        from sympy import Matrix
    aug = Matrix([list(row) + [v] for row, v in zip(A, b)])
    raug, pivot = aug.rref()
    nindep = len(pivot)
    return raug[:nindep, :-1], raug[:nindep, -1]


def linear_exprs(A, x, b=None, rref=False, Matrix=None):
    """
    returns Ax - b

    A: matrix_like of numbers
        of shape (len(b), len(x))
    x: iterable of symbols
    b: array_like of numbers (default: None)
        when None, assume zeros of length len(x)
    Matrix: class
        When ``rref == Ture``: A matrix class which supports slicing,
        and methods ``dot`` and ``rref``. Defaults to sympy.Matrix
    rref: bool (default: False)
        calculate the reduced row echelon form of (A | -b)
    """
    if b is None:
        b = [0]*len(x)
    if rref:
        rA, rb = linear_rref(A, b, Matrix)
        return [lhs - rhs for lhs, rhs in zip(rA.dot(x), rb)]
    else:
        return [sum([x0*x1 for x0, x1 in zip(row, x)]) - v
                for row, v in zip(A, b)]
