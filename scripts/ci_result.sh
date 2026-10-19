#!/usr/bin/env bash
# Decide the CI run's one required check from the results of the jobs it needs.
#
# A needed job passes when it succeeded, or when it was skipped because the paths
# it tests did not change. The jobs named as arguments must have succeeded: a
# skipped path filter would otherwise pass a run that tested nothing.
#
# Reads:
#   NEEDS_JSON  the workflow's toJSON(needs)
#
# Usage: ci_result.sh [job that must succeed]...

set -euo pipefail

needs="${NEEDS_JSON:?NEEDS_JSON is required}"

if [ "$(jq 'length' <<< "${needs}")" -eq 0 ]; then
  echo "ci-result: no needed jobs to judge" >&2
  exit 1
fi

jq -r 'to_entries[] | "\(.key): \(.value.result)"' <<< "${needs}"

must_succeed=$(printf '%s\n' "$@" | jq -R . | jq -s .)
bad=$(jq -r --argjson must "${must_succeed}" '
  to_entries[]
  | select(
      .value.result != "success"
      and (.value.result != "skipped" or (.key as $k | $must | index($k)))
    )
  | .key' <<< "${needs}")

if [ -n "${bad}" ]; then
  echo "ci-result: these jobs did not pass: ${bad//$'\n'/ }" >&2
  exit 1
fi
echo "ci-result: every needed job passed or was rightly skipped"
