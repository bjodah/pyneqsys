---
title: 'pyneqsys: Solve symbolically defined systems of non-linear equations numerically'
tags:
  - systems of non-linear equations
  - symbolic derivation
  - symbolic transformations
authors:
 - name: Björn Dahlgren
   orcid: 0000-0003-0596-0222
   affiliation: 1
affiliations:
 - name: KTH Royal Institute of Technology
   index: 1
date: 8 January 2018
bibliography: paper.bib
---

# Summary
Solving systems of non-linear equations numceriaclly is a common task in scientific modelling
work. Many software libraries have the capability to these kinds of systems, however, each
require the slightly different forms of input. In addition, it is often important that the
user formulates the system in a manner which is suitable for the numerical algorithm. Coming
up with an effective formulation is often an iterative process, and it is therefore of great
value if transformations of systems could be performed symbolically.

*pyneqsys* offers a common interface to a handful of solvers. The user may also use it for
its facilities for working with the system symbolically. Having a symbolic representation
allows *pyneqsys* to automatically derive the Jacobian matrix, which is a task which is
laborious and a source of error when performed by hand. By relying on a computer algebra system,
*pyneqsys* allows the user to apply e.g. variable transformations and also generate represenations
in e.g. LaTeX, MathML etc. By default SymPy [@Meurer2017] is used as the symbolic backend, but the
users may choose to use another library if they so wish.

Adapting *pyneqsys* to use new thrid party solvers is straightforward and some example solvers are
provided with the library. Together with the ability to perform variable transformations symbolically
*pyneqsys* allows the users to write code for their problems *once*, which greatly lowers the burden
of validation and also speeds-up the iterative nature of finding the best method for solving the problem.


# Features
- Unified interface to the KINSOL solver from SUNDIALS [@hindmarsh2005sundials],
  SciPy's solvers [@jones_scipy_2001], levmar [@lourakis04LM], NLEQ2 [@weimann1991family] and mpmath [@mpmath].
- Convenince methods for solving and plotting solutions.
- Automatic derivation of the Jacobian matrix.
- Symbolic variable transformations in the system.
- Symbolic removal of linearly dependent equations by rewriting in reduced row echelon form.
- Carrying over the solution as initial guess in parameter variations.
- Facility for defining meta-algorithms (e.g. solve the system for one formulation first, and
  refine the solution by solving it as another formulation).
- Solve non-linear systems containing conditional equations (different equations govering different
  domains).

# References