#!/usr/bin/env bash
# Re-release an existing tag's source under a new version.
#
#   make promote FROM=v0.6.90b420.dev726 TO=0.6.90b420
#   bash scripts/promote_release.sh v0.6.90b420.dev726 0.6.90b420
#   DRY_RUN=1 bash scripts/promote_release.sh v0.6.90b420.dev726 0.6.90b420
#
# release-candidate.yml calls a tag a promotion when the parent of its commit is
# another tag's commit and nothing but pyproject.toml and uv.lock differs between
# the two. It then copies the source release's notes instead of generating new
# ones. This builds exactly that commit: the version bump alone, on top of FROM.
#
# The work happens in a throwaway worktree, so the checkout you are sitting in is
# never touched, and only the tag is pushed. The commit is deliberately not
# pushed to main: its parent is FROM, which main has usually moved past.
set -euo pipefail

from="${1:-}"
to="${2:-}"
if [ -z "$from" ] || [ -z "$to" ]; then
  echo "usage: $0 <FROM-tag> <TO-version>" >&2
  echo "   eg: $0 v0.6.90b420.dev726 0.6.90b420" >&2
  exit 1
fi
case "$from" in
  v*) ;;
  *) echo "promote: FROM must be a tag starting with v (got '$from')" >&2; exit 1 ;;
esac
case "$to" in
  v*) echo "promote: TO must be a bare version, no leading v (got '$to')" >&2; exit 1 ;;
esac

cd "$(git rev-parse --show-toplevel)"

git fetch -q origin "refs/tags/${from}:refs/tags/${from}" 2>/dev/null || true
git rev-parse -q --verify "refs/tags/${from}" >/dev/null 2>&1 \
  || { echo "promote: no such tag: ${from}" >&2; exit 1; }
if git ls-remote --exit-code --tags origin "refs/tags/v${to}" >/dev/null 2>&1; then
  echo "promote: v${to} already exists on origin; pick another version" >&2
  exit 1
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/lilbee-promote.XXXXXX")
cleanup() { git worktree remove --force "$work" >/dev/null 2>&1 || rm -rf "$work"; }
trap cleanup EXIT
git worktree add -q --detach "$work" "refs/tags/${from}"

cur=$(awk -F'"' '/^version *= */ { print $2; exit }' "$work/pyproject.toml")
[ -n "$cur" ] || { echo "promote: could not read the version from ${from}" >&2; exit 1; }
[ "$cur" != "$to" ] || { echo "promote: ${from} is already version ${to}" >&2; exit 1; }

perl -pi -e 's/^version = "\Q'"$cur"'\E"$/version = "'"$to"'"/' \
  "$work/pyproject.toml" "$work/uv.lock"

# The promotion is only detected when these two files are the whole diff, so
# check it here rather than finding out from a release that generated its notes
# from scratch.
changed=$(git -C "$work" diff --name-only | sort | tr '\n' ' ')
if [ "$changed" != "pyproject.toml uv.lock " ]; then
  echo "promote: expected only pyproject.toml and uv.lock to change, got: ${changed}" >&2
  exit 1
fi

echo "promote: ${from} (${cur}) -> v${to}"
git -C "$work" --no-pager diff --stat

if [ -n "${DRY_RUN:-}" ]; then
  echo "promote: DRY_RUN set, so nothing was committed, tagged, or pushed."
  exit 0
fi

git -C "$work" add pyproject.toml uv.lock
git -C "$work" commit -q -m "Release ${to}"
git -C "$work" tag "v${to}"
git -C "$work" push origin "v${to}"

echo "promote: pushed v${to}; release-candidate.yml is building it as a promotion of ${from}."
echo "promote: when that pipeline is green, run 'make release-promote TAG=v${to}'."
