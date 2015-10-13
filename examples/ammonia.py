#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function)


import sympy as sp
import numpy as np

from pyneqsys import SymbolicSys
from pyneqsys.symbolic import linear_exprs


def main(init_conc='1e-7,1e-7,1e-7,1,55.5',
         lnKa=-21.28, lnKw=-36.25,
         savetxt='None', verbose=False,
         rref=False, charge=False):
    # H+, OH- NH4+, NH3, H2O
    iHp, iOHm, iNH4p, iNH3, iH2O = init_conc = map(float, init_conc.split(','))
    lHp, lOHm, lNH4p, lNH3, lH2O = x = sp.symarray('x', 5)
    Hp, OHm, NH4p, NH3, H2O = map(sp.exp, x)
    # lHp + lOHm - lH2O - lnKw,
    # lHp + lNH3 - lNH4p - lnKa,
    coeffs = [[1, 1, 0, 0, -1], [1, 0, -1, 1, 0]]
    vals = [lnKw, lnKa]
    lp = linear_exprs(coeffs, x, vals, rref=rref)
    f = lp + [
        Hp + OHm + 4*NH4p + 3*NH3 + 2*H2O - (
            iHp + iOHm + 4*iNH4p + 3*iNH3 + 2*iH2O),  # H
        NH4p + NH3 - (iNH4p + iNH3),  # N
        OHm + H2O - (iOHm + iH2O)
    ]
    if charge:
        f += [Hp - OHm + NH4p - (iHp - iOHm + iNH4p)]

    neqsys = SymbolicSys(x, f)
    x, sol = neqsys.solve_scipy([0]*5)
    if verbose:
        print(np.exp(x), sol)
    else:
        print(np.exp(x))
    assert sol.success

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
