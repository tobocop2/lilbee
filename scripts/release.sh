#!/usr/bin/env bash
# Cut a beta release: bump the trailing bNNN, commit, tag, and push from main.
# release-candidate.yml builds the artifacts, publishes to PyPI, and creates the
# pre-release with generated notes. Once that pipeline is green run
# `make release-promote` to rewrite the notes as headings and mark it latest.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" = "main" ] || { echo "release: must be on main (on $branch)" >&2; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "release: tracked changes present; commit or stash first" >&2; exit 1; }
git fetch -q origin main
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] \
  || { echo "release: main is not in sync with origin/main" >&2; exit 1; }

cur=$(awk -F'"' '/^version *= */ { print $2; exit }' pyproject.toml)
case "$cur" in
  *b[0-9]*) ;;
  *) echo "release: version '$cur' has no beta (bNNN) segment to bump" >&2; exit 1;;
esac
next="${cur%b*}b$(( ${cur##*b} + 1 ))"
tag="v${next}"
echo "release: $cur -> $next ($tag)"

perl -pi -e 's/^version = "\Q'"$cur"'\E"$/version = "'"$next"'"/' pyproject.toml uv.lock

git add pyproject.toml uv.lock
git commit -q -m "Release ${next}"
git tag "$tag"
git push origin main
git push origin "$tag"

echo "release: pushed ${tag}; release-candidate.yml is building."
echo "release: when the PyPI publish is green, run 'make release-promote'."
