#!/usr/bin/env bash
# Rerun the release-candidate cells that failed, so the candidate corrects itself.
#
# A candidate builds for hours across ~30 cells, and a cell that dies from an
# infra flake (runner lost communication, no space left on device) strands the
# release short of promotion until a person notices.
#
# `gh run rerun --failed` is the whole mechanism. It reruns the failed jobs plus
# the jobs skipped behind them and leaves the successful cells alone, which is
# what both failure modes need:
#
#   - A soft cell (every cell in build-gpu-executables.yml is continue-on-error)
#     carries conclusion=failure while the run stays green. Nothing is skipped,
#     so only that cell reruns; its own release_tag step attaches the missing
#     asset, and verify-release re-fires when the run completes again.
#   - A hard cell takes the run red, which skips attach-prerelease and every
#     dispatch job. Rerunning the cell alone would leave them skipped forever,
#     because a per-job rerun does not re-queue dependents. `--failed` picks the
#     stranded cascade up with the cell.
#
# Every dispatch job on a tag build shares one `if` (push + refs/tags/v), so a
# skipped job there always means "stranded", never "skipped by design". That is
# what makes --failed safe to point at this workflow.
#
# The retry is blind: it does not read logs to guess whether a failure was a
# flake or a defect. Signature matching against GitHub's error text is a list
# that goes stale, and a defect just fails again and stops at the attempt bound.
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

# --paginate is correct here: the filter emits one name per line, so pages
# concatenate. It would be wrong with a jq count, which runs per page.
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
