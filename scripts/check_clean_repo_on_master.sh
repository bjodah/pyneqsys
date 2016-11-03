#!/bin/bash
if [[ $(git rev-parse --abbrev-ref HEAD) != master ]]; then
    echo "We are not on the master branch. Aborting..."
    exit 1
fi
if [[ ! -z $(git status -s) ]]; then
    echo "'git status' show there are some untracked/uncommited changes. Aborting..."
    exit 1
fi
if grep -e "^v" CHANGES.rst; then
    if ! grep -e "^$1"; then
        >&2 echo "CHANGES.rst does not contain an entry for: $1"
        exit 1
    fi
else
    >&2 echo "CHANGES.rst does not start with v*"
    exit 1
fi
