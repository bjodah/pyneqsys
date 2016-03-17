# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import pytest

from .test_core import _test_neqsys_no_params

try:
    import ipopt
except ImportError:
    ipopt = None


@pytest.mark.skipif(ipopt is None, reason='ipopt package unavailable')
@pytest.mark.xfail
def test_ipopt():
    _test_neqsys_no_params('ipopt')
