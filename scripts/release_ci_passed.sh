#!/usr/bin/env bash
# Print ci_passed=true when CI already linted and tested this exact commit.
# The verdict is the only line on stdout; the reason goes to stderr.
#
# A CI run whose paths filter skipped every job still concludes success, so a
# run counts only when its lint job ran and succeeded. Fail-soft: any lookup
# that fails prints ci_passed=false, and the release runs lint and test itself.
#
# Usage: release_ci_passed.sh <owner/repo> <ref>

set -uo pipefail

repo="$1"
ref="${2#refs/tags/}"
ref="${ref#refs/heads/}"

verdict() {
  echo "ci_passed=$1"
  exit 0
}

if ! sha=$(gh api "repos/${repo}/commits/${ref}" --jq .sha) || [ -z "${sha}" ]; then
  echo "couldn't resolve ${ref} -> SHA; will run lint + test" >&2
  verdict false
fi

if ! ids=$(gh api "repos/${repo}/actions/workflows/ci.yml/runs?head_sha=${sha}&per_page=100" \
    --jq '.workflow_runs[] | select(.conclusion == "success") | .id'); then
  echo "couldn't list the CI runs for ${sha}; will run lint + test" >&2
  verdict false
fi

for id in ${ids}; do
  if linted=$(gh api "repos/${repo}/actions/runs/${id}/jobs?per_page=100" \
      --jq '[.jobs[] | select(.name == "lint" and .conclusion == "success")] | length') \
      && [ "${linted:-0}" -gt 0 ]; then
    echo "${ref} -> ${sha}; CI run ${id} linted and tested it" >&2
    verdict true
  fi
done

echo "${ref} -> ${sha}; no successful CI run linted and tested it" >&2
verdict false
