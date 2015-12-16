v0.3.0
======
- NeqSys.solve() refactored
- NeqSys.solve_scipy and NeqSys.solve_nleq2 was made private
- Added ChainedNeqSys
- In NeqSys.solve() arg "solver" may now be None -> $NEQSYS_SOLVER
- NeqSys.plot_series changed signature
- New methods: NeqSys.plot_series_residuals(_internal)

v0.2.1
======
- Fixed bug in SymbolicSys jacobian evaluation.
- Use pyodesys mechanism to dynamically use sympy/symengine/pysym for pyneqsys.symbolic

v0.2.0
======
- Use of of pre-/post-processors

v0.1.4
======
- Added ConditionalNeqSys

v0.1.3
======
- Added convenince methods: solve_series and plot_series to NeqSys

v0.1.2
======
- argument order in symbolic.linear_exprs changed to more natural A, x, b for Ax = b
- TransformedSys now also takes a `expr_tranf` keyword. (see new examples/chem_equil_ammonia.ipynb)

v0.1.1
======
- provisional support for symengine

v0.1
====
- support for scipy
