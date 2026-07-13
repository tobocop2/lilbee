#!/usr/bin/env bash
# Cut a release: bump pyproject to the next final-form version, commit, tag, and
# push from main. Versions carry no pre-release markers -- a build's maturity
# lives in the release channels (packaging/channels.json), not in its version
# string, so the exact same artifacts can move dev -> beta -> stable untouched.
# release-candidate.yml builds everything once and enters the dev channel;
# `make promote TAG=... CHANNEL=beta|stable` moves it up.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" = "main" ] || { echo "release: must be on main (on $branch)" >&2; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "release: tracked changes present; commit or stash first" >&2; exit 1; }
git fetch -q origin main
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] \
  || { echo "release: main is not in sync with origin/main" >&2; exit 1; }

cur=$(awk -F'"' '/^version *= */ { print $2; exit }' pyproject.toml)
# A final-form version bumps its last segment (0.6.90 -> 0.6.91). A version
# still carrying markers from the old scheme graduates to its base version
# (0.6.90b420.dev719 -> 0.6.90), which PEP 440 orders above all its pre-releases.
base=$(printf '%s' "$cur" | sed -E 's/(a|b|rc)[0-9].*$//; s/\.dev[0-9]+$//')
if [ "$base" != "$cur" ]; then
  next="$base"
else
  # 10# keeps a zero-padded counter out of bash's octal interpretation.
  next="${cur%.*}.$(( 10#${cur##*.} + 1 ))"
fi
tag="v${next}"
echo "release: $cur -> $next ($tag)"

perl -pi -e 's/^version = "\Q'"$cur"'\E"$/version = "'"$next"'"/' pyproject.toml uv.lock

git add pyproject.toml uv.lock
git commit -q -m "Release ${next}"
git tag "$tag"
git push origin main
git push origin "$tag"

echo "release: pushed ${tag}; release-candidate.yml is building and will enter the dev channel."
echo "release: once verify-release is green, run 'make promote TAG=${tag} CHANNEL=beta',"
echo "release: and after the beta soak, 'make promote TAG=${tag} CHANNEL=stable'."
