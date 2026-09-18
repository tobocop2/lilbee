#!/usr/bin/env bash
# Print the release-candidate cells worth rerunning, one job name per line.
#
#   RUN_ID=1234567 bash scripts/release_dropped_cells.sh
#
# A cell that fails drops out with conclusion `failure`. A cell past its
# timeout-minutes drops out with conclusion `cancelled`, and so does every cell
# of a run a person cancelled, including the run's own conclusion. So the two
# are told apart by the job's own annotation, where the runner records the time
# limit it exceeded. An unrecognized annotation heals nothing, which is the
# behavior a person cancelling a release expects. An annotation list that cannot
# be read is unknown, not clean, and exits non-zero.
#
# Environment:
#   RUN_ID  release-candidate run to read (required)
#   REPO    owner/name (default: tobocop2/lilbee)

set -uo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
REPO="${REPO:-tobocop2/lilbee}"

TIME_LIMIT_ANNOTATION="The job has exceeded the maximum execution time"

exceeded_its_time_limit() {  # job-id
  local hit
  if ! hit=$(gh api "repos/${REPO}/check-runs/$1/annotations" \
    -q ".[] | select(.message | startswith(\"${TIME_LIMIT_ANNOTATION}\")) | .message"); then
    echo "release_dropped_cells: cannot read the annotations for job $1." >&2
    echo "The dropped cells are unknown. Grant checks: read, or rerun when the API answers." >&2
    exit 1
  fi
  [ -n "${hit}" ]
}

# --paginate is safe here: the filter emits one line per job, not a count.
jobs=$(gh api "repos/${REPO}/actions/runs/${RUN_ID}/jobs?per_page=100" \
  --paginate -q '.jobs[] | [.conclusion, .id, .name] | @tsv') || exit 1

# Read on fd 3: the gh call inside the loop would take the job list off stdin.
while IFS=$'\t' read -r conclusion job_id name <&3; do
  case "${conclusion}" in
    failure) echo "${name}" ;;
    cancelled)
      if exceeded_its_time_limit "${job_id}"; then echo "${name}"; fi
      ;;
  esac
done 3<<< "${jobs}"
