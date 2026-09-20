#!/usr/bin/env bash
# Fail when an executable the release cannot ship without is missing from a
# directory of downloaded artifacts.
#
#   bash scripts/assert_required_binaries.sh bins
#
# The required set is derived from release.yml's build matrix: a cell carrying
# `soft: true` may drop out for any reason, including a timeout, and every other
# cell may not. Reading the matrix keeps one source of truth for which cell is
# required.

set -euo pipefail

dir="${1:?directory holding the downloaded executables is required}"
workflow="${2:-.github/workflows/release.yml}"

# yq reads the matrix as YAML, so a re-indented or reordered cell still counts.
# It ships with the ubuntu-latest runner image that attach-prerelease runs on.
required=$(yq -r '
  (.jobs.build.strategy.matrix.include // [])[]
  | select(.soft != true)
  | .asset_name
' "${workflow}")

if [ -z "${required}" ]; then
  echo "assert_required_binaries: read no required cell out of ${workflow}." >&2
  echo "assert_required_binaries: the matrix shape changed; update this derivation." >&2
  exit 1
fi

missing=""
while IFS= read -r asset; do
  [ -f "${dir}/${asset}" ] || missing="${missing}  ${asset}"$'\n'
done <<< "${required}"

if [ -n "${missing}" ]; then
  echo "assert_required_binaries: these executables never reached ${dir}:" >&2
  printf '%s' "${missing}" >&2
  echo "assert_required_binaries: read the build cell for each one and rerun it." >&2
  exit 1
fi

echo "every required executable is present in ${dir}"
