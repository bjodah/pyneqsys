from .. import SymbolicSys
from .test_core import f

import pysym  # minimal alternative to SymPy


def test_pysym_SymbolicSys_from_callback():
    ss = SymbolicSys.from_callback(
        f, 2, 1,
        lambdify=pysym.Lambdify,
        lambdify_unpack=False,
        symarray=pysym.symarray,
        Matrix=pysym.Matrix)

    x, sol = ss.solve('scipy', [1, 0], [3])
    assert sol.success
    assert abs(x[0] - 0.8411639) < 2e-7
    assert abs(x[1] - 0.1588361) < 2e-7
