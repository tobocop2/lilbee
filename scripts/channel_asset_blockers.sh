#!/usr/bin/env bash
# Print the channel artifacts that can no longer reach a release, one line each.
#
#   REPO=o/n TAG=v1.2.3 RC_RUN_ID=123 bash scripts/channel_asset_blockers.sh lilbee.flatpakref ...
#
# Each channel artifact is built by one publish workflow, and the release
# candidate starts that workflow from one dispatch job. An artifact that is
# still missing can only arrive while one of the two is live. It is blocked when
#   - the publish workflow ran for the tag and has finished, or
#   - no such run exists and the dispatch job in the release candidate ended
#     skipped, failed or cancelled, so nothing started the workflow.
# A run that has not finished, and a dispatch that succeeded and has not shown
# its run yet, are not blocked: the caller keeps waiting. Without RC_RUN_ID only
# the first rule applies. The release assets are read after the verdicts, so an
# upload that lands while the verdicts are read is never reported as blocked. A
# lookup that fails exits non-zero.
#
# Environment:
#   REPO        owner/name (required)
#   TAG         release tag (required)
#   RC_RUN_ID   release-candidate run that dispatched the publishers (optional)

set -euo pipefail

REPO="${REPO:?REPO is required}"
TAG="${TAG:?TAG is required}"
RC_RUN_ID="${RC_RUN_ID:-}"

# artifact -> publish workflow | its run-name prefix | the dispatch job that starts it
producer() {
  case "$1" in
    *.flatpakref) echo "publish-flatpak.yml|Publish Flatpak|dispatch-fanout" ;;
    lilbee-linux-x86_64.snap) echo "publish-packages.yml|Publish Packages|dispatch-fanout" ;;
    lilbee-linux-x86_64-cu125.snap) echo "publish-cuda-packages.yml|Publish CUDA Packages|dispatch-cuda" ;;
    lilbee-linux-x86_64-rocm.snap) echo "publish-rocm-packages.yml|Publish ROCm Packages|dispatch-rocm" ;;
    lilbee-compat-linux-x86_64.snap) echo "publish-compat-packages.yml|Publish compat packages|dispatch-compat" ;;
    lilbee-compat-linux-x86_64-cu124.snap) echo "publish-compat-cuda-packages.yml|Publish compat CUDA packages|dispatch-compat-cuda" ;;
    *) return 1 ;;
  esac
}

dispatch_jobs='[]'
if [ -n "${RC_RUN_ID}" ]; then
  dispatch_jobs=$(gh run view "${RC_RUN_ID}" --repo "${REPO}" --json jobs \
    | jq -c '[.jobs[] | {name, conclusion}]')
fi

blocked=()
for asset in "$@"; do
  spec=$(producer "${asset}") || continue
  IFS='|' read -r workflow prefix dispatch <<< "${spec}"

  run=$(gh run list --repo "${REPO}" --workflow="${workflow}" --limit 30 \
    --json databaseId,status,conclusion,displayTitle \
    | jq -c --arg title "${prefix} ${TAG}" '[.[] | select(.displayTitle == $title)] | first // empty')
  if [ -n "${run}" ]; then
    if [ "$(jq -r .status <<< "${run}")" = "completed" ]; then
      blocked+=("${asset}: ${workflow} run $(jq -r .databaseId <<< "${run}") finished ($(jq -r .conclusion <<< "${run}")) without it")
    fi
    continue
  fi

  conclusion=$(jq -r --arg job "${dispatch}" '[.[] | select(.name == $job)] | first | .conclusion // empty' <<< "${dispatch_jobs}")
  case "${conclusion}" in
    skipped | failure | cancelled)
      blocked+=("${asset}: ${dispatch} ${conclusion} in the release candidate, so ${workflow} never ran") ;;
  esac
done

[ "${#blocked[@]}" -gt 0 ] || exit 0
present=$(gh release view "${TAG}" --repo "${REPO}" --json assets --jq '.assets[].name')
for line in "${blocked[@]}"; do
  grep -qxF "${line%%: *}" <<< "${present}" || echo "${line}"
done
