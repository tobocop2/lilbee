#!/usr/bin/env bash
# Print the release.yml build-matrix assets, one name per line.
#
#   bash scripts/release_required_assets.sh            # the required ones
#   bash scripts/release_required_assets.sh --soft     # the droppable ones
#
# A cell carrying `soft: true` is droppable and every other cell is required.
# `soft` is declared once, in release.yml, and read once, here, so an asset
# cannot be droppable for one gate and required by another.
#
# Two rules divide the callers:
#   - a gate that decides whether the RELEASE is complete requires only the
#     required set, so a droppable cell can never block a tag;
#   - a publisher that SHIPS an asset requires that asset whatever this says,
#     because it cannot publish without it, and its failure reaches only the
#     channel that ships it.

set -euo pipefail

want_soft=false
if [ "${1:-}" = "--soft" ]; then
  want_soft=true
  shift
fi
workflow="${1:-.github/workflows/release.yml}"

# yq reads the matrix as YAML, so a re-indented or reordered cell still counts.
# It ships with the ubuntu-latest runner image the release jobs run on.
cells=$(yq -r '
  (.jobs.build.strategy.matrix.include // [])[]
  | [(.soft == true), .asset_name]
  | @tsv
' "${workflow}")

if [ -z "${cells}" ]; then
  echo "release_required_assets: read no build cell out of ${workflow}." >&2
  echo "release_required_assets: the matrix shape changed; update this derivation." >&2
  exit 1
fi

selected=$(awk -F'\t' -v want="${want_soft}" '$1 == want { print $2 }' <<< "${cells}")

if [ -z "${selected}" ]; then
  echo "release_required_assets: ${workflow} has no cell with soft=${want_soft}." >&2
  echo "release_required_assets: the matrix shape changed; update this derivation." >&2
  exit 1
fi

printf '%s\n' "${selected}"
