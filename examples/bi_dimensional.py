#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sympy as sp
import numpy as np

from pyneqsys import SymbolicSys


def main(init_a=0, init_b=0, savetxt='None'):
    x = sp.symarray('x', 2)
    f = [x[0] + (x[0] - x[1])**3/2 - 1,
         (x[1] - x[0])**3/2 + x[1]]
    neqsys = SymbolicSys(x, f)
    x, sol = neqsys.solve_scipy([init_a, init_b])
    if savetxt != 'None':
        np.savetxt(x, savetxt)
    else:
        print(sol)

if __name__ == '__main__':
    try:
        import argh
        argh.dispatch_command(main)
    except ImportError:
        import sys
        if len(sys.argv) > 1:
            import warnings
            warnings.warn("Ignoring parameters run "
                          "'pip install --user argh' to fix.")
        main()
