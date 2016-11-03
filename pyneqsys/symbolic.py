# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from itertools import chain

import numpy as np
from sym import Backend
from sym.util import banded_jacobian, check_transforms

from .core import NeqSys, _ensure_3args


def _map2(cb, iterable):
    if cb is None:  # identity function is assumed
        return iterable
    else:
        return map(cb, iterable)


def _map2l(cb, iterable):  # Py2 type of map in Py3
    return list(_map2(cb, iterable))


class SymbolicSys(NeqSys):
    """
    Parameters
    ----------
    x : iterable of Symbols
    exprs : iterable of expressions for f
    params : iterable of Symbols (optional)
        list of symbols appearing in exprs which are parameters
    jac : ImmutableMatrix or bool
        If True:
            calculate jacobian from exprs
        If False:
            do not compute jacobian (numeric approximation)
        If ImmutableMatrix:
            user provided expressions for the jacobian
    backend : str or sym.Backend
        See documentation of `sym.Backend \
<https://pythonhosted.org/sym/sym.html#sym.backend.Backend>`_.
    \*\*kwargs:
        See :py:class:`pyneqsys.core.NeqSys`

    Examples
    --------
    >>> import sympy as sp
    >>> e = sp.exp
    >>> x = x0, x1 = sp.symbols('x:2')
    >>> params = a, b = sp.symbols('a b')
    >>> neqsys = SymbolicSys(x, [a*(1 - x0), b*(x1 - x0**2)], params)
    >>> xout, sol = neqsys.solve('scipy', [-10, -5], [1, 10])
    >>> print(xout)
    [ 1.  1.]
    >>> print(neqsys.get_jac()[0, 0])
    -a

    Notes
    -----
    When using SymPy as backend a limited number of unknowns is supported.
    The reason is that (currently) ``sympy.lambdify`` has an upper limit on
    number of arguments.

    """

    def __init__(self, x, exprs, params=(), jac=True, backend=None, **kwargs):
        self.x = x
        self.exprs = exprs
        self.params = params
        self._jac = jac
        self.be = Backend(backend)
        self.nf, self.nx = len(exprs), len(x)  # needed by get_*_callback
        self.band = kwargs.get('band', None)  # needed by get_*_callback
        super(SymbolicSys, self).__init__(self.nf, self.nx,
                                          self._get_f_callback(),
                                          self._get_j_callback(),
                                          **kwargs)

    @classmethod
    def from_callback(cls, cb, nx, nparams=0, **kwargs):
        """ Generate a SymbolicSys instance from a callback"""
        be = Backend(kwargs.pop('backend', None))
        x = be.real_symarray('x', nx)
        p = be.real_symarray('p', nparams)
        try:
            exprs = cb(x, p, be)
        except TypeError:
            exprs = _ensure_3args(cb)(x, p, be)
        return cls(x, exprs, p, backend=be, **kwargs)

    def get_jac(self):
        """ Return the jacobian of the expressions """
        if self._jac is True:
            if self.band is None:
                f = self.be.Matrix(self.nf, 1, self.exprs)
                _x = self.be.Matrix(self.nx, 1, self.x)
                return f.jacobian(_x)
            else:
                # Banded
                return self.be.Matrix(banded_jacobian(
                    self.exprs, self.x, *self.band))
        elif self._jac is False:
            return False
        else:
            return self._jac

    def _get_f_callback(self):
        args = list(chain(self.x, self.params))
        cb = self.be.Lambdify(args, self.exprs)

        def f(x, params):
            return cb(np.concatenate((x, params), axis=-1))
        return f

    def _get_j_callback(self):
        cb = self.be.Lambdify(list(chain(self.x, self.params)), self.get_jac())

        def j(x, params):
            return cb(np.concatenate((x, params), axis=-1))
        return j


class TransformedSys(SymbolicSys):
    """ A system which transforms the equations and variables internally

    Can be used to reformulate a problem in a numerically more stable form.

    Parameters
    ----------
    x: iterable of variables
    exprs: iterable of expressions
         expressions to find root for (untransformed)
    transf: iterable of pairs of expressions
        forward, backward transformed instances of x
    params: iterable of symbols
    post_adj: callable (default: None)
        to tweak expression after transformation
    \*\*kwargs:
        keyword arguments passed onto :class:`SymbolicSys`
    """

    def __init__(self, x, exprs, transf, params=(), post_adj=None, **kwargs):
        self.fw, self.bw = zip(*transf)
        check_transforms(self.fw, self.bw, x)
        exprs = [e.subs(zip(x, self.fw)) for e in exprs]
        super(TransformedSys, self).__init__(
            x, _map2l(post_adj, exprs), params,
            pre_processors=[lambda xarr, params: (self.bw_cb(xarr), params)],
            post_processors=[lambda xarr, params: (self.fw_cb(xarr), params)],
            **kwargs)
        self.fw_cb = self.be.Lambdify(x, self.fw)
        self.bw_cb = self.be.Lambdify(x, self.bw)

    @classmethod
    def from_callback(cls, cb, transf_cbs, nx, nparams=0, pre_adj=None,
                      **kwargs):
        """ Generate a TransformedSys instance from a callback

        Parameters
        ----------
        cb: callable
        transf_cbs: pair or iterable of pairs of callables
        nx: int
        nparams: int
        pre_adj: callable
        \*\*kwargs: passed onto TransformedSys
        """
        be = Backend(kwargs.pop('backend', None))
        x = be.real_symarray('x', nx)
        p = be.real_symarray('p', nparams)
        try:
            transf = [(transf_cbs[idx][0](xi),
                       transf_cbs[idx][1](xi))
                      for idx, xi in enumerate(x)]
        except TypeError:
            transf = zip(_map2(transf_cbs[0], x), _map2(transf_cbs[1], x))
        try:
            exprs = cb(x, p, be)
        except TypeError:
            exprs = _ensure_3args(cb)(x, p, be)
        return cls(x, _map2l(pre_adj, exprs), transf, p, backend=be, **kwargs)


def linear_rref(A, b, Matrix=None, S=None):
    """ Transform a linear system to reduced row-echelon form

    Transforms both the matrix and right-hand side of a linear
    system of equations to reduced row echelon form

    Parameters
    ----------
    A: Matrix-like
        iterable of rows
    b: iterable

    Returns
    -------
    A', b' - transformed versions

    """
    if Matrix is None:
        from sympy import Matrix
    if S is None:
        from sympy import S
    mat_rows = [_map2l(S, list(row) + [v]) for row, v in zip(A, b)]
    aug = Matrix(mat_rows)
    raug, pivot = aug.rref()
    nindep = len(pivot)
    return raug[:nindep, :-1], raug[:nindep, -1]


def linear_exprs(A, x, b=None, rref=False, Matrix=None):
    """
    returns Ax - b

    Parameters
    ----------
    A: matrix_like of numbers
        of shape (len(b), len(x))
    x: iterable of symbols
    b: array_like of numbers (default: None)
        when None, assume zeros of length len(x)
    Matrix: class
        When ``rref == True``: A matrix class which supports slicing,
        and methods ``dot`` and ``rref``. Defaults to sympy.Matrix
    rref: bool (default: False)
        calculate the reduced row echelon form of (A | -b)

    Returns
    -------
    A list of the elements in the resulting column vector
    """
    if b is None:
        b = [0]*len(x)
    if rref:
        rA, rb = linear_rref(A, b, Matrix)
        return [lhs - rhs for lhs, rhs in zip(rA.dot(x), rb)]
    else:
        return [sum([x0*x1 for x0, x1 in zip(row, x)]) - v
                for row, v in zip(A, b)]
