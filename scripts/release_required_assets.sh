#!/usr/bin/env bash
# Print the assets a release cannot ship without, one name per line.
#
#   bash scripts/release_required_assets.sh [.github/workflows/release.yml]
#
# The set is release.yml's build matrix minus the cells carrying `soft: true`.
# Every gate that decides whether an asset may be absent reads it from here, so
# an asset cannot be droppable for one gate and required by another.

set -euo pipefail

workflow="${1:-.github/workflows/release.yml}"

# yq reads the matrix as YAML, so a re-indented or reordered cell still counts.
# It ships with the ubuntu-latest runner image the release jobs run on.
required=$(yq -r '
  (.jobs.build.strategy.matrix.include // [])[]
  | select(.soft != true)
  | .asset_name
' "${workflow}")

if [ -z "${required}" ]; then
  echo "release_required_assets: read no required cell out of ${workflow}." >&2
  echo "release_required_assets: the matrix shape changed; update this derivation." >&2
  exit 1
fi

printf '%s\n' "${required}"
