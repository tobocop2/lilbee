#!/usr/bin/env bash
# Refuse a flake sources file whose pins are malformed: a version, and for each
# system an asset name and a sha256 of 64 hex digits. `nix flake check
# --no-build` does not read a fixed-output hash, so it passes a bad one.
#
# Usage: check_flake_pins.sh <sources file>...

set -uo pipefail

if [ "$#" -eq 0 ]; then
  echo "check_flake_pins.sh: no sources file to check" >&2
  exit 2
fi

bad=0
for f in "$@"; do
  if ! problems=$(jq -r '
      (if (.version | type) == "string" and .version != "" then empty else "version is missing" end),
      (if (.systems | type) == "object" then
         .systems | to_entries[]
         | (if (.value.asset | type) == "string" and .value.asset != "" then empty
            else "\(.key): asset is missing" end),
           (if (.value.sha256 | type) == "string" and (.value.sha256 | test("^[0-9a-f]{64}$")) then empty
            else "\(.key): sha256 is not 64 hex digits" end)
       else "systems is not an object" end)' "${f}" 2>&1); then
    problems="not valid JSON: ${problems}"
  fi
  if [ -n "${problems}" ]; then
    printf '%s: %s\n' "${f}" "${problems//$'\n'/; }" >&2
    bad=1
  else
    echo "${f}: ok"
  fi
done
exit "${bad}"
