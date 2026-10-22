#!/usr/bin/env bash
# Print the release build cells as one line of JSON, for the build matrix.
#
#   bash scripts/release_plan_cells.sh [only_asset] [cells-file]
#
# With no asset the output holds every cell in .github/release-cells.yml.
# With an asset name it holds that one cell, and an unknown name fails, so a
# typo cannot turn into a green run that built nothing.

set -euo pipefail

only="${1:-}"
cells="${2:-.github/release-cells.yml}"

# yq prints each cell as one JSON document; jq filters and joins them. This
# subset reads the same under the yq on the runner image and under yq's jq
# wrapper.
plan=$(yq -r '.[] | @json' "${cells}" \
  | jq -sc --arg only "${only}" '[.[] | select($only == "" or .asset_name == $only)]')

if [ "${plan}" = "[]" ]; then
  echo "release_plan_cells: no build cell in ${cells} matches '${only}'." >&2
  exit 1
fi

printf '%s\n' "${plan}"
