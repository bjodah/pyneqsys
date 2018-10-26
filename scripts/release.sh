#!/bin/bash -xeu
# Usage:
#
#    $ ./scripts/release.sh v1.2.3 GITHUB_USER GITHUB_REPO
#

if [[ $1 != v* ]]; then
    echo "Argument does not start with 'v'"
    exit 1
fi
VERSION=${1#v}
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
MPLBACKEND=Agg ./scripts/run_tests.sh
env ${PKG_UPPER}_RELEASE_VERSION=v$VERSION python setup.py sdist
if [[ -e ./scripts/generate_docs.sh ]]; then
    env ${PKG_UPPER}_RELEASE_VERSION=v$VERSION ./scripts/generate_docs.sh  # $4 ${5:-$PKG} v$VERSION
fi

# All went well, add a tag and push it.
git tag -a v$VERSION -m v$VERSION
git push
git push --tags
twine upload dist/${PKG}-$VERSION.tar.gz

set +x
echo ""
echo "    You may now create a new github release at with the tag \"v$VERSION\" and name "
echo "    it \"${PKG}-${VERSION}\", (don't foreget to manually attach the new .tar.gz"
echo "    file from the ./dist/ directory). Here is a link:"
echo "        https://github.com/$2/${3:-$PKG}/releases/new "
echo "    Then run:"
echo ""
echo "        $ ./scripts/post_release.sh $1 <myserver.example.com>"
echo ""
