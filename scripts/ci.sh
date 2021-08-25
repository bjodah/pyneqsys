#!/bin/bash -xeu
PKG_NAME=${1:-${CI_REPO##*/}}
if [[ "$DRONE_BRANCH" =~ ^v[0-9]+.[0-9]?* ]]; then
    eval export ${PKG_NAME^^}_RELEASE_VERSION=\$DRONE_BRANCH
    echo ${DRONE_BRANCH} | tail -c +2 > __conda_version__.txt
fi
python3 -m pip install --user --upgrade pip
python3 -m pip install --user --upgrade pip

git archive -o /tmp/$PKG_NAME.zip HEAD  # test pip installable zip (symlinks break)
python3 -m pip install /tmp/$PKG_NAME.zip

python3 setup.py sdist  # test pip installable sdist (checks MANIFEST.in)
(cd dist/; python3 -m pip install $PKG_NAME-$(python3 ../setup.py --version).tar.gz)
(cd /; python3 -m pytest --pyargs $PKG_NAME)

python3 -m pip install --user -e .[all] pysym pykinsol
python3 -m pip install --user -e .[all] pysym pykinsol
PYTHONPATH=$(pwd) PYTHON=python3 ./scripts/run_tests.sh
python3 -m pip uninstall -y fastcache
PYTHONPATH=$(pwd) PYTHON=python3 ./scripts/run_tests.sh
PYTHONPATH=$(pwd) PYTHON=python3 ./scripts/run_tests.sh --cov $PKG_NAME --cov-report html
./scripts/coverage_badge.py htmlcov/ htmlcov/coverage.svg
! grep "DO-NOT-MERGE!" -R . --exclude ci.sh

./scripts/render_notebooks.sh
./scripts/generate_docs.sh

# PATH=/opt/miniconda3/bin:$PATH conda config --add channels bjodah  # sym
# PATH=/opt/miniconda3/bin:$PATH conda update -c defaults --quiet conda-build
# PATH=/opt/miniconda3/bin:$PATH conda-build conda-recipe
