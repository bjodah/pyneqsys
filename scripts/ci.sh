#!/bin/bash -xeu
PKG_NAME=${1:-${CI_REPO##*/}}
if [[ "$CI_BRANCH" =~ ^v[0-9]+.[0-9]?* ]]; then
    eval export ${PKG_NAME^^}_RELEASE_VERSION=\$CI_BRANCH
    echo ${CI_BRANCH} | tail -c +2 > __conda_version__.txt
fi
python2.7 setup.py sdist
(cd dist/; python2.7 -m pip install $PKG_NAME-$(python ../setup.py --version).tar.gz)
(cd dist/; python3.4 -m pip install $PKG_NAME-$(python ../setup.py --version).tar.gz)
(cd /; python2.7 -m pytest --pyargs $PKG_NAME)
(cd /; python3.4 -m pytest --pyargs $PKG_NAME)
python2.7 -m pip install --user -e .[all] pysym pykinsol git+https://github.com/bjodah/cyipopt.git
python3.4 -m pip install --user -e .[all] pysym pykinsol git+https://github.com/bjodah/cyipopt.git
PYTHONPATH=$(pwd) PYTHON=python2.7 ./scripts/run_tests.sh
PYTHONPATH=$(pwd) PYTHON=python3.4 ./scripts/run_tests.sh --cov $PKG_NAME --cov-report html
./scripts/coverage_badge.py htmlcov/ htmlcov/coverage.svg
! grep "DO-NOT-MERGE!" -R . --exclude ci.sh
