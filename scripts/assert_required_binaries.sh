#!/usr/bin/env bash
# Fail when an executable the release cannot ship without is missing from a
# directory of downloaded artifacts.
#
#   bash scripts/assert_required_binaries.sh bins
#
# scripts/release_required_assets.sh owns which assets those are.

set -euo pipefail

dir="${1:?directory holding the downloaded executables is required}"
workflow="${2:-.github/workflows/release.yml}"

required=$(bash "$(dirname "$0")/release_required_assets.sh" "${workflow}")

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
