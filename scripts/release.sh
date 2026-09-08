#!/usr/bin/env bash
# Cut a beta release: bump the trailing counter (the .devNNN when the version has
# one, else the bNNN), commit, tag, and push from main.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" = "main" ] || { echo "release: must be on main (on $branch)" >&2; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "release: tracked changes present; commit or stash first" >&2; exit 1; }
git fetch -q origin main
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] \
  || { echo "release: main is not in sync with origin/main" >&2; exit 1; }

cur=$(awk -F'"' '/^version *= */ { print $2; exit }' pyproject.toml)
# The counter must END the version: 0.7.0b1.post1 passes a "contains bN" test
# and then fails inside the arithmetic below, after the branch and sync checks.
case "$cur" in
  *.dev[0-9]|*.dev[0-9][0-9]*|*b[0-9]|*b[0-9][0-9]*) ;;
  *) echo "release: version '$cur' does not end in a bNNN or .devNNN counter to bump" >&2; exit 1;;
esac
# Bump the last numeric segment: the dev counter when the version carries one
# (0.6.90b420.dev710 -> .dev711), otherwise the beta counter (0.6.66b507 -> b508).
# 10# keeps a zero-padded counter out of bash's octal interpretation.
case "$cur" in
  *.dev[0-9]*) next="${cur%.dev*}.dev$(( 10#${cur##*.dev} + 1 ))" ;;
  *)           next="${cur%b*}b$(( 10#${cur##*b} + 1 ))" ;;
esac
tag="v${next}"
echo "release: $cur -> $next ($tag)"

perl -pi -e 's/^version = "\Q'"$cur"'\E"$/version = "'"$next"'"/' pyproject.toml uv.lock

git add pyproject.toml uv.lock
git commit -q -m "Release ${next}"
git tag "$tag"
# One push: a rejected main (someone landed between the fetch above and here)
# must not leave the tag published against a commit that never reached main.
git push --atomic origin main "$tag"

echo "release: pushed ${tag}. The pipeline takes it from here: it builds, publishes"
echo "release: every channel, retries a leg that flakes, and promotes the tag itself."
echo "release: a red 'Watch ${tag}' run is the only thing that wants your attention."
