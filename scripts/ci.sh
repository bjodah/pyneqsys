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
for pypkg in pycompilation pycodeexport pykinsol sym symcxx pysym levmar; do
    case $pypkg in
        levmar)
            pypkg_fqn="git+https://github.com/bjodah/levmar#egg=levmar"             
            ;;
        symcxx)
            pypkg_fqn="git+https://github.com/bjodah/symcxx#egg=symcxx"
            ;;
        pysym)
            pypkg_fqn="git+https://github.com/bjodah/pysym#egg=pysym"
            ;;
        sym)
            pypkg_fqn="git+https://github.com/bjodah/sym@jun21#egg=sym"
            ;;
        pycompilation)
            pypkg_fqn="git+https://github.com/bjodah/pycompilation@master#egg=pycompilation"
            ;;
        pycodeexport)
            pypkg_fqn="git+https://github.com/bjodah/pycodeexport@master#egg=pycodeexport"
            ;;
        pykinsol)
            pypkg_fqn="git+https://github.com/bjodah/pykinsol@jan25#egg=pykinsol"
            ;;
        *)
            pypkg_fqn=$pypkg
            ;;
    esac
    python -m pip install ${INSTALL_PIP_FLAGS:-} $pypkg_fqn
    python -m pytest --pyargs $pypkg
done

python3 -m pip install /tmp/$PKG_NAME.zip

python3 setup.py sdist  # test pip installable sdist (checks MANIFEST.in)
(cd dist/; python3 -m pip install $PKG_NAME-$(python3 ../setup.py --version).tar.gz)
(cd /; python3 -m pytest --pyargs $PKG_NAME)

python3 -m pip install .[all]
PYTHONPATH=$(pwd) PYTHON=python3 ./scripts/run_tests.sh --cov $PKG_NAME --cov-report html
./scripts/coverage_badge.py htmlcov/ htmlcov/coverage.svg
! grep "DO-NOT-MERGE!" -R . --exclude ci.sh

./scripts/render_notebooks.sh
./scripts/generate_docs.sh

# PATH=/opt/miniconda3/bin:$PATH conda config --add channels bjodah  # sym
# PATH=/opt/miniconda3/bin:$PATH conda update -c defaults --quiet conda-build
# PATH=/opt/miniconda3/bin:$PATH conda-build conda-recipe
