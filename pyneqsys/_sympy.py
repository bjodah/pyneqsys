from sympy.printing.latex import LatexPrinter


class NeqSysTexPrinter(LatexPrinter):

    def _print_list(self, lst):
        return r"\\".join(['0 = %s' % self._print(e) for e in lst])
