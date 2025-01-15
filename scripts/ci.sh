#!/bin/bash
set -xeuo pipefail
PKG_NAME=${1:-${CI_REPO_NAME##*/}}
if [[ "$DRONE_BRANCH" =~ ^v[0-9]+.[0-9]?* ]]; then
    eval export ${PKG_NAME^^}_RELEASE_VERSION=\$DRONE_BRANCH
    echo ${DRONE_BRANCH} | tail -c +2 > __conda_version__.txt
fi

SUNDIALS_ROOT=$(compgen -G "/opt-3/sundials-6.*-release")
if [ ! -e $SUNDIALS_ROOT/include/sundials/sundials_config.h ]; then
    >&2 echo "No functional sundials install at SUNDIALS_ROOT?: $SUNDIALS_ROOT"
    exit 1
fi
export CPATH=$SUNDIALS_ROOT/include
export LIBRARY_PATH=$SUNDIALS_ROOT/lib
export LD_LIBRARY_PATH=$SUNDIALS_ROOT/lib
source /opt-3/cpython-v3.*-apt-deb/bin/activate

git archive -o /tmp/$PKG_NAME.zip HEAD  # test pip installable zip (symlinks break)
python3 -m pip install /tmp/$PKG_NAME.zip

python3 setup.py sdist  # test pip installable sdist (checks MANIFEST.in)
(cd dist/; python3 -m pip install $PKG_NAME-$(python3 ../setup.py --version).tar.gz)
(cd /; python3 -m pytest --pyargs $PKG_NAME)

python3 -m pip install pykinsol "git+https://github.com/bjodah/pysym#egg=pysym"
python3 -m pip install .[all] pysym pykinsol
PYTHONPATH=$(pwd) PYTHON=python3 ./scripts/run_tests.sh --cov $PKG_NAME --cov-report html
./scripts/coverage_badge.py htmlcov/ htmlcov/coverage.svg
! grep "DO-NOT-MERGE!" -R . --exclude ci.sh

./scripts/render_notebooks.sh
./scripts/generate_docs.sh

# PATH=/opt/miniconda3/bin:$PATH conda config --add channels bjodah  # sym
# PATH=/opt/miniconda3/bin:$PATH conda update -c defaults --quiet conda-build
# PATH=/opt/miniconda3/bin:$PATH conda-build conda-recipe
