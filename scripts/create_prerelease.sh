#!/usr/bin/env bash
# Create the candidate pre-release shell idempotently, carrying the source
# release notes over when this tag promotes one. Args: tag version repo [from].
set -euo pipefail

tag="$1"
version="$2"
repo="$3"
promoted_from="${4:-}"

gh release view "${tag}" --repo "${repo}" >/dev/null 2>&1 && exit 0
notes=$(mktemp)
if [ -n "${promoted_from}" ] \
  && gh release view "${promoted_from}" --repo "${repo}" --json body -q .body > "${notes}.src" 2>/dev/null; then
  {
    echo "Promoted from [${promoted_from}](https://github.com/${repo}/releases/tag/${promoted_from}): the same source, released as ${version}."
    echo
    cat "${notes}.src"
  } > "${notes}"
else
  echo "Release candidate building; executables attach here as each one finishes." > "${notes}"
fi
gh release create "${tag}" --repo "${repo}" --prerelease --notes-file "${notes}"
