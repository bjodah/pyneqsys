#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# PYTHON_ARGCOMPLETE_OK
# Pass --help flag for help on command-line interface

from __future__ import (absolute_import, division, print_function)


import sympy as sp
import numpy as np

from pyneqsys.symbolic import SymbolicSys, linear_exprs


def main(init_conc_molar='1e-7,1e-7,1e-7,1,55.5', lnKa=-21.28, lnKw=-36.25, verbose=False,
         rref=False, charge=False, solver='scipy'):
    """ Aqueous ammonia protolysis (chemical species: H+, OH-, NH4+, NH3, H2O)

    In this example will look at how we can use pyneqsys to solve the coupled chemical
    equilibria governing the protolysis of ammonia in water (together with water's
    auto-protolysis).
    """

    # We will need intial values for our chemical species, using the order given above:
    iHp, iOHm, iNH4p, iNH3, iH2O = map(float, init_conc_molar.split(','))

    # We will also need SymPy symbols. We will define an equal number of symbols
    # echo denoting the natural logarithm of the concentration of our species:
    lHp, lOHm, lNH4p, lNH3, lH2O = x = sp.symarray('x', 5)

    # The concentrations of each specie are then a SymPy expression:
    Hp, OHm, NH4p, NH3, H2O = map(sp.exp, x)

    # We have two chemical equilibria:
    #
    #    H+ + OH- = H2O
    #    H+ + NH3 = NH4+
    #
    # these two equations give two linear equations (they are linear with
    # respect to logarithm of concentration, not the actual concentration):
    #
    #    lHp + lOHm - lH2O - lnKw = 0
    #    lHp + lNH3 - lNH4p - lnKa = 0
    #
    # Describing the two above equations in terms of a matrix "coeffs":
    coeffs = [[1, 1, 0, 0, -1], [1, 0, -1, 1, 0]]

    # and a "right-hand-side" vals:
    vals = [lnKw, lnKa]

    # we can formulate SymPy expressions:
    lp = linear_exprs(coeffs, x, vals, rref=rref)
    # note the keyword-argument ``rref``, when True, it asks SymPy
    # to rewrite the system in "reduced row echelon form"

    # We need 3 more equations to be able to solve our problem, by
    # writing down the conservation laws of our atom types we can
    # get those additional equations
    conserv_H = Hp + OHm + 4*NH4p + 3*NH3 + 2*H2O - (
        iHp + iOHm + 4*iNH4p + 3*iNH3 + 2*iH2O)
    conserv_N = NH4p + NH3 - (iNH4p + iNH3)
    conserv_O = OHm + H2O - (iOHm + iH2O)

    eqs = lp + [conserv_H, conserv_N, conserv_O]
    if charge:   # we can add a conservation law for charge as well, but it is linearly dependent
        eqs += [Hp - OHm + NH4p - (iHp - iOHm + iNH4p)]

    # From our SymPy symbols and equations we can now create a ``SymbolicSys`` instance:
    neqsys = SymbolicSys(x, eqs)

    # To solve our non-linear system of equations we need to pick a guess:
    guess = [0]*5   # ln(concentration / molar) == 0  =>  concentration == 1 molar
    # And call the ``solve`` method:
    x, sol = neqsys.solve(guess, solver=solver)

    # Finally we print the concentrations by applying the exponential function to the logarithmic values:
    if verbose:
        print(np.exp(x), sol)
    else:
        print(np.exp(x))
    assert sol.success


if __name__ == '__main__':  # <--- this checks it the file was invoked from the command line instead of imported from
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
