#!/bin/bash
# Clone gh-pages, copy in the page from main, run publish.py with the given arguments, push.
# Retries so that several scoring runs finishing at once don't clobber each other.
# Usage (from the repo root on main, with GH_TOKEN set): scoring/update_site.sh <publish.py args after <site>>
set -euo pipefail
for attempt in 1 2 3 4 5; do
    rm -rf site_build
    git clone --quiet --depth 1 --branch gh-pages "https://x-access-token:$GH_TOKEN@github.com/$GITHUB_REPOSITORY" site_build
    sed "s#__REPO__#$GITHUB_REPOSITORY#" site/index.html > site_build/index.html
    python scoring/publish.py site_build "$@"
    cd site_build
    git add -A
    git -c user.name="golf-bot" -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
        commit --quiet -m "Update leaderboard ($*)" || { cd ..; exit 0; }
    if git push --quiet origin gh-pages; then cd ..; exit 0; fi
    cd ..
    echo "push conflict, retrying ($attempt)" >&2
    sleep $((RANDOM % 10 + 5))
done
exit 1
