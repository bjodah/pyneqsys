from sympy.printing.latex import LatexPrinter


class NeqSysTexPrinter(LatexPrinter):
    """ SymPy printer used to allow pretty-printing of SymbolicSys in Jupyter notebooks. """

    def _print_list(self, lst):
        return r"\\".join(['0 = %s' % self._print(e) for e in lst])
