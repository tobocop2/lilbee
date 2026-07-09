#!/usr/bin/env bash
# Cut a beta release: bump the trailing counter (the .devNNN when the version has
# one, else the bNNN), commit, tag, and push from main.
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
git push origin main
git push origin "$tag"

echo "release: pushed ${tag}; release-candidate.yml is building."
echo "release: when the PyPI publish is green, run 'make release-promote'."
