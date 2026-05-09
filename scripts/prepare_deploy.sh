#!/bin/bash
touch doc/_build/html/.nojekyll
cp LICENSE doc/_build/html/.nojekyll
git config --global user.name "drone"
git config --global user.email "drone@nohost.com"
mkdir -p deploy/public_html/branches/"${CI_BRANCH}" deploy/script_queue
cp -r dist/* htmlcov/ examples/ doc/_build/html/ deploy/public_html/branches/"${CI_BRANCH}"/
if bash -c '[[ "$CI_BRANCH" == "master" ]]'; then
    sed -e "s/\$1/public_html\/branches\/${CI_BRANCH}\/html/" -e "s/\$2/bjodah/" -e "s/\$3/$CI_REPO/" -e 's/$4/gh-pages/' <scripts/dir_to_branch.sh >deploy/script_queue/gh-pages.sh
    chmod +x deploy/script_queue/gh-pages.sh
    if [ -e benchmarks/ ]; then
        cat <<EOF>deploy/script_queue/run_benchmark.sh
source /etc/profile
cd ~/benchmarks/
asv run -k -e >asv-run.log
asv publish>asv-publish.log
EOF
        chmod +x deploy/script_queue/run_benchmark.sh
        cp -r benchmarks/ deploy/
    fi
fi
