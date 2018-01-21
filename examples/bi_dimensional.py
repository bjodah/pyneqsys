#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# PYTHON_ARGCOMPLETE_OK
# Pass --help flag for help on command-line interface

import sympy as sp
import numpy as np

from pyneqsys.symbolic import SymbolicSys


def solve(guess_a, guess_b, power, solver='scipy'):
    """ Constructs a pyneqsys.symbolic.SymbolicSys instance and returns from its ``solve`` method. """
    # The problem is 2 dimensional so we need 2 symbols
    x = sp.symbols('x:2', real=True)
    # There is a user specified parameter ``p`` in this problem:
    p = sp.Symbol('p', real=True, negative=False, integer=True)
    # Our system consists of 2-non-linear equations:
    f = [x[0] + (x[0] - x[1])**p/2 - 1,
         (x[1] - x[0])**p/2 + x[1]]
    # We construct our ``SymbolicSys`` instance by passing variables, equations and parameters:
    neqsys = SymbolicSys(x, f, [p])  # (this will derive the Jacobian symbolically)

    # Finally we solve the system using user-specified ``solver`` choice:
    return neqsys.solve([guess_a, guess_b], [power], solver=solver)


def main(guess_a=1., guess_b=0., power=3, savetxt='None', verbose=False):
    """
    Example demonstrating how to solve a system of non-linear equations defined as SymPy expressions.

    The example shows how a non-linear problem can be given a command-line interface which may be
    preferred by end-users who are not familiar with Python.
    """
    x, sol = solve(guess_a, guess_b, power)  # see function definition above
    assert sol.success
    if savetxt != 'None':
        np.savetxt(x, savetxt)
    else:
        if verbose:
            print(sol)
        else:
            print(x)


if __name__ == '__main__':  # are we running from the command line (or are we being imported from)?
    try:
        import argh
        argh.dispatch_command(main, output_file=None)
    except ImportError:
        import sys
        if len(sys.argv) > 1:
            import warnings
            warnings.warn("Ignoring parameters run "
                          "'pip install --user argh' to fix.")
        main()
