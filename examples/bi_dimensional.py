#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sympy as sp
import numpy as np

from pyneqsys import SymbolicSys


def solve(init_a, init_b, power, solver='scipy'):
    x = sp.symbols('x:2', real=True)
    p = sp.Symbol('p', real=True, negative=False, integer=True)
    f = [x[0] + (x[0] - x[1])**p/2 - 1,
         (x[1] - x[0])**p/2 + x[1]]
    neqsys = SymbolicSys(x, f, [p])
    return neqsys.solve(solver, [init_a, init_b], [power])


def main(init_a=1., init_b=0., power=3, savetxt='None', verbose=False):
    """
    Demonstrate how to solve a system of non-linear eqautions
    defined as SymPy expressions.
    """
    x, sol = solve(init_a, init_b, power)
    assert sol.success
    if savetxt != 'None':
        np.savetxt(x, savetxt)
    else:
        if verbose:
            print(sol)
        else:
            print(x)


if __name__ == '__main__':
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
