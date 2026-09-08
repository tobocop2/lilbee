#!/usr/bin/env bash
# Rerun the release-candidate cells that failed, so the candidate corrects itself.
#
# `gh run rerun --failed` reruns the failed jobs plus the jobs skipped behind
# them. A soft cell (build-gpu-executables.yml is all continue-on-error) reruns
# alone and re-attaches its asset; a hard cell takes the run red and strands the
# dispatch cascade, which a per-job rerun would leave skipped forever.
#
# Run it by hand against a finished candidate:
#   RUN_ID=1234567 bash scripts/release_selfheal.sh
#
# Environment:
#   RUN_ID        release-candidate run to heal (required)
#   REPO          owner/name (default: tobocop2/lilbee)
#   MAX_ATTEMPTS  attempts before a failure is called a defect (default 2)

set -euo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
REPO="${REPO:-tobocop2/lilbee}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"   # see AGENTS.md on the release attempt bound
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/stdout}"

attempt=$(gh api "repos/${REPO}/actions/runs/${RUN_ID}" -q .run_attempt)
conclusion=$(gh api "repos/${REPO}/actions/runs/${RUN_ID}" -q .conclusion)

# --paginate is safe here: the filter emits one name per line, not a count.
failed=$(gh api "repos/${REPO}/actions/runs/${RUN_ID}/jobs?per_page=100" \
  --paginate -q '.jobs[] | select(.conclusion == "failure") | .name')

{
  echo "## Release self-heal"
  echo
  echo "Run [${RUN_ID}](https://github.com/${REPO}/actions/runs/${RUN_ID}), attempt ${attempt}, concluded ${conclusion}."
  echo
} >> "${SUMMARY}"

if [ -z "${failed}" ]; then
  echo "no failed cells; nothing to heal" | tee -a "${SUMMARY}"
  exit 0
fi

{
  echo "Failed cells:"
  echo
  echo "${failed}" | awk '{ print "- " $0 }'
  echo
} >> "${SUMMARY}"

if [ "${attempt}" -ge "${MAX_ATTEMPTS}" ]; then
  echo "attempt ${attempt} reached the bound of ${MAX_ATTEMPTS}; not retrying again" | tee -a "${SUMMARY}"
  echo "These cells failed twice. Read the logs before rerunning: a second failure is a defect until proven otherwise." >> "${SUMMARY}"
  exit 1
fi

echo "rerunning the failed cells and anything stranded behind them"
gh run rerun "${RUN_ID}" --repo "${REPO}" --failed
echo "Retried as attempt $((attempt + 1))." >> "${SUMMARY}"
