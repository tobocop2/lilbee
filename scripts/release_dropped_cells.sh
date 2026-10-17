#!/usr/bin/env bash
# Print the release-candidate cells worth rerunning, one job name per line.
#
#   RUN_ID=1234567 bash scripts/release_dropped_cells.sh
#
# A cell that fails drops out with conclusion `failure`. A cell past its
# timeout-minutes drops out with conclusion `cancelled`, and so does every cell
# of a run a person cancelled. The runner records a person's cancel in the
# job's annotations ("The run was canceled by @user." or "...canceled
# forcefully by @user."), and the time limit it exceeded in another. A run with
# any person-cancelled cell prints nothing: `gh run rerun --failed` reruns every
# cancelled cell, so one failed cell would otherwise bring the cancelled ones
# back. An unrecognized annotation heals nothing, which is the behavior a
# person cancelling a release expects. An annotation list that cannot be read is
# unknown, not clean, and exits non-zero.
#
# Environment:
#   RUN_ID  release-candidate run to read (required)
#   REPO    owner/name (default: tobocop2/lilbee)

set -uo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
REPO="${REPO:-tobocop2/lilbee}"

TIME_LIMIT_ANNOTATION="The job has exceeded the maximum execution time"
CANCEL_ANNOTATION="The run was canceled"

annotations() {  # job-id -> the annotation messages, one per line
  local messages
  if ! messages=$(gh api "repos/${REPO}/check-runs/$1/annotations" -q '.[] | .message'); then
    echo "release_dropped_cells: cannot read the annotations for job $1." >&2
    echo "The dropped cells are unknown. Grant checks: read, or rerun when the API answers." >&2
    exit 1
  fi
  printf '%s\n' "${messages}"
}

# --paginate is safe here: the filter emits one line per job, not a count.
jobs=$(gh api "repos/${REPO}/actions/runs/${RUN_ID}/jobs?per_page=100" \
  --paginate -q '.jobs[] | [.conclusion, .id, .name] | @tsv') || exit 1

dropped=""
cancelled_by_person=false
# Read on fd 3: the gh call inside the loop would take the job list off stdin.
while IFS=$'\t' read -r conclusion job_id name <&3; do
  case "${conclusion}" in
    failure) dropped="${dropped}${name}"$'\n' ;;
    cancelled)
      notes=$(annotations "${job_id}") || exit 1
      if grep -q "^${CANCEL_ANNOTATION}" <<< "${notes}"; then
        cancelled_by_person=true
      elif grep -q "^${TIME_LIMIT_ANNOTATION}" <<< "${notes}"; then
        dropped="${dropped}${name}"$'\n'
      fi
      ;;
  esac
done 3<<< "${jobs}"

if [ "${cancelled_by_person}" = true ]; then
  echo "release_dropped_cells: a person cancelled this run; nothing is healed." >&2
  exit 0
fi
printf '%s' "${dropped}"
