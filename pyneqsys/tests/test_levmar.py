# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import pytest

from .test_core import (
    _test_neqsys_params, _test_neqsys_no_params, _test_fail
)

try:
    import levmar
except ImportError:
    levmar = None


@pytest.mark.skipif(levmar is None, reason='levmar package unavailable')
def test_levmar_neqsys_no_params():
    _test_neqsys_no_params('levmar', eps1=1e-10, eps2=1e-10, eps3=1e-10)


@pytest.mark.skipif(levmar is None, reason='levmar package unavailable')
def test_levmar_neqsys_params():
    _test_neqsys_params('levmar', eps1=1e-10, eps2=1e-10, eps3=1e-10)


@pytest.mark.skipif(levmar is None, reason='levmar package unavailable')
def test_levmar_fail():
    _test_fail('levmar')
