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
    """ Symbolically defined system of non-linear equations.

    This object is analogous to :class:`pyneqsys.NeqSys` but instead of
    providing a callable, the user provides symbolic expressions.

    Parameters
    ----------
    x : iterable of Symbols
    exprs : iterable of expressions for ``f``
    params : iterable of Symbols (optional)
        list of symbols appearing in exprs which are parameters
    jac : ImmutableMatrix or bool
        If ``True``:
            - Calculate Jacobian from ``exprs``.
        If ``False``:
            - Do not compute Jacobian (numeric approximation).
        If ImmutableMatrix:
            - User provided expressions for the Jacobian.
    backend : str or sym.Backend
        See documentation of `sym.Backend \
<https://pythonhosted.org/sym/sym.html#sym.backend.Backend>`_.
    module : str
        ``module`` keyword argument passed to ``backend.Lambdify``.
    \\*\\*kwargs:
        See :py:class:`pyneqsys.core.NeqSys`.

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
    When using SymPy as the backend, a limited number of unknowns is supported.
    The reason is that (currently) ``sympy.lambdify`` has an upper limit on
    number of arguments.

    """

    def __init__(self, x, exprs, params=(), jac=True, backend=None, **kwargs):
        self.x = x
        self.exprs = exprs
        self.params = params
        self._jac = jac
        self.be = Backend(backend)
        self.nf, self.nx = len(exprs), len(x)  # needed by get_*_cb
        self.band = kwargs.get('band', None)  # needed by get_*_cb
        self.module = kwargs.pop('module', 'numpy')
        super(SymbolicSys, self).__init__(self.nf, self.nx,
                                          self._get_f_cb(),
                                          self._get_j_cb(),
                                          **kwargs)

    @classmethod
    def from_callback(cls, cb, nx=None, nparams=None, **kwargs):
        """ Generate a SymbolicSys instance from a callback.

        Parameters
        ----------
        cb : callable
            Should have the signature ``cb(x, p, backend) -> list of exprs``.
        nx : int
            Number of unknowns, when not given it is deduced from ``kwargs['names']``.
        nparams : int
            Number of parameters, when not given it is deduced from ``kwargs['param_names']``.

        \\*\\*kwargs :
            Keyword arguments passed on to :class:`SymbolicSys`. See also :class:`pyneqsys.NeqSys`.

        Examples
        --------
        >>> symbolicsys = SymbolicSys.from_callback(lambda x, p, be: [
        ...     x[0]*x[1] - p[0],
        ...     be.exp(-x[0]) + be.exp(-x[1]) - p[0]**-2
        ... ], 2, 1)
        ...

        """
        if kwargs.get('x_by_name', False):
            if 'names' not in kwargs:
                raise ValueError("Need ``names`` in kwargs.")
            if nx is None:
                nx = len(kwargs['names'])
            elif nx != len(kwargs['names']):
                raise ValueError("Inconsistency between nx and length of ``names``.")
        if kwargs.get('par_by_name', False):
            if 'param_names' not in kwargs:
                raise ValueError("Need ``param_names`` in kwargs.")
            if nparams is None:
                nparams = len(kwargs['param_names'])
            elif nparams != len(kwargs['param_names']):
                raise ValueError("Inconsistency between ``nparam`` and length of ``param_names``.")

        if nparams is None:
            nparams = 0

        if nx is None:
            raise ValueError("Need ``nx`` of ``names`` together with ``x_by_name==True``.")
        be = Backend(kwargs.pop('backend', None))
        x = be.real_symarray('x', nx)
        p = be.real_symarray('p', nparams)
        _x = dict(zip(kwargs['names'], x)) if kwargs.get('x_by_name', False) else x
        _p = dict(zip(kwargs['param_names'], p)) if kwargs.get('par_by_name', False) else p
        try:
            exprs = cb(_x, _p, be)
        except TypeError:
            exprs = _ensure_3args(cb)(_x, _p, be)
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

    def _get_f_cb(self):
        args = list(chain(self.x, self.params))
        kw = dict(module=self.module, dtype=object if self.module == 'mpmath' else None)
        try:
            cb = self.be.Lambdify(args, self.exprs, **kw)
        except TypeError:
            cb = self.be.Lambdify(args, self.exprs)

        def f(x, params):
            return cb(np.concatenate((x, params), axis=-1))
        return f

    def _get_j_cb(self):
        args = list(chain(self.x, self.params))
        kw = dict(module=self.module, dtype=object if self.module == 'mpmath' else None)
        try:
            cb = self.be.Lambdify(args, self.get_jac(), **kw)
        except TypeError:
            cb = self.be.Lambdify(args, self.get_jac())

        def j(x, params):
            return cb(np.concatenate((x, params), axis=-1))
        return j

    _use_symbol_latex_names = True

    def _repr_latex_(self):  # pretty printing in Jupyter notebook
        from ._sympy import NeqSysTexPrinter
        if self.latex_names and (self.latex_param_names if len(self.params) else True):
            pretty = {s: n for s, n in chain(
                zip(self.x, self.latex_names) if self._use_symbol_latex_names else [],
                zip(self.params, self.latex_param_names)
            )}
        else:
            pretty = {}

        return '$%s$' % NeqSysTexPrinter(dict(symbol_names=pretty)).doprint(self.exprs)


class TransformedSys(SymbolicSys):
    """ A system which transforms the equations and variables internally

    Can be used to reformulate a problem in a numerically more stable form.

    Parameters
    ----------
    x : iterable of variables
    exprs : iterable of expressions
         Expressions to find root for (untransformed).
    transf : iterable of pairs of expressions
        Forward, backward transformed instances of x.
    params : iterable of symbols
    post_adj : callable (default: None)
        To tweak expression after transformation.
    \\*\\*kwargs :
        Keyword arguments passed onto :class:`SymbolicSys`.

    """
    _use_symbol_latex_names = False  # symbols have been transformed

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
        cb : callable
            Should have the signature ``cb(x, p, backend) -> list of exprs``.
            The callback ``cb`` should return *untransformed* expressions.
        transf_cbs : pair or iterable of pairs of callables
            Callables for forward- and backward-transformations. Each
            callable should take a single parameter (expression) and
            return a single expression.
        nx : int
            Number of unkowns.
        nparams : int
            Number of parameters.
        pre_adj : callable, optional
            To tweak expression prior to transformation. Takes a
            sinlge argument (expression) and return a single argument
            rewritten expression.
        \\*\\*kwargs :
            Keyword arguments passed on to :class:`TransformedSys`. See also
            :class:`SymbolicSys` and :class:`pyneqsys.NeqSys`.

        Examples
        --------
        >>> import sympy as sp
        >>> transformed = TransformedSys.from_callback(lambda x, p, be: [
        ...     x[0]*x[1] - p[0],
        ...     be.exp(-x[0]) + be.exp(-x[1]) - p[0]**-2
        ... ], (sp.log, sp.exp), 2, 1)
        ...


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
    A : Matrix-like
        Iterable of rows.
    b : iterable

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
    """ Returns Ax - b

    Parameters
    ----------
    A : matrix_like of numbers
        Of shape (len(b), len(x)).
    x : iterable of symbols
    b : array_like of numbers (default: None)
        When ``None``, assume zeros of length ``len(x)``.
    Matrix : class
        When ``rref == True``: A matrix class which supports slicing,
        and methods ``dot`` and ``rref``. Defaults to ``sympy.Matrix``.
    rref : bool
        Calculate the reduced row echelon form of (A | -b).

    Returns
    -------
    A list of the elements in the resulting column vector.

    """
    if b is None:
        b = [0]*len(x)
    if rref:
        rA, rb = linear_rref(A, b, Matrix)
        return [lhs - rhs for lhs, rhs in zip(rA.dot(x), rb)]
    else:
        return [sum([x0*x1 for x0, x1 in zip(row, x)]) - v
                for row, v in zip(A, b)]
