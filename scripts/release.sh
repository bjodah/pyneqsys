#!/bin/bash -xeu
# Usage:
#
#    $ ./scripts/release.sh v1.2.3 ~/anaconda2/bin myserver.example.com GITHUB_USER GITHUB_REPO
#

if [[ $1 != v* ]]; then
    echo "Argument does not start with 'v'"
    exit 1
fi
VERSION=${1#v}
CONDA_PATH=$2
SERVER=$3
find . -type f -iname "*.pyc" -exec rm {} +
find . -type f -iname "*.o" -exec rm {} +
find . -type f -iname "*.so" -exec rm {} +
find . -type d -name "__pycache__" -exec rmdir {} +
./scripts/check_clean_repo_on_master.sh
cd $(dirname $0)/..
# PKG will be name of the directory one level up containing "__init__.py" 
PKG=$(find . -maxdepth 2 -name __init__.py -print0 | xargs -0 -n1 dirname | xargs basename)
! grep --include "*.py" "will_be_missing_in='$VERSION'" -R $PKG/  # see deprecation()
PKG_UPPER=$(echo $PKG | tr '[:lower:]' '[:upper:]')
./scripts/run_tests.sh
env ${PKG_UPPER}_RELEASE_VERSION=v$VERSION python setup.py sdist
if [[ -e ./scripts/generate_docs.sh ]]; then
    env ${PKG_UPPER}_RELEASE_VERSION=v$VERSION ./scripts/generate_docs.sh  # $4 ${5:-$PKG} v$VERSION
fi
for CONDA_PY in 2.7 3.4 3.5; do
    for CONDA_NPY in 1.11; do
        continue  # we build the conda recipe on another host for now..
        PATH=$CONDA_PATH:$PATH ./scripts/build_conda_recipe.sh v$VERSION --python $CONDA_PY --numpy $CONDA_NPY
    done
done

# All went well, add a tag and push it.
git tag -a v$VERSION -m v$VERSION
git push
git push --tags
twine upload dist/${PKG}-$VERSION.tar.gz

set +x
echo ""
echo "    You may now create a new github release at with the tag \"v$VERSION\", here is a link:"
echo "        https://github.com/$4/${5:-$PKG}/releases/new "
echo "    name the release \"${PKG}-${VERSION}\", and don't foreget to manually attach the file:"
echo "        $(openssl sha256 $(pwd)/dist/${PKG}-${VERSION}.tar.gz)"
echo "    Then run:"
echo ""
echo "        $ ./scripts/post_release.sh $1 $SERVER $4"
echo ""
