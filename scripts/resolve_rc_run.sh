#!/usr/bin/env bash
# Print the id of the release-candidate run that built a tag.
#
#   REPO=o/n TAG=v1.2.3 SHA=<tag commit> bash scripts/resolve_rc_run.sh
#
# A run is picked by its original identity. `createdAt` never changes, while
# `run_started_at` moves to the newest attempt, so re-running an old run would
# put it ahead of the right one. A run is a candidate only when it built the
# commit the tag points at now and was not cancelled. An explicit RC_RUN_ID is
# checked by the same two rules instead of being trusted.
#
# Environment:
#   REPO        owner/name (required)
#   TAG         release tag (required)
#   SHA         commit the tag points at (required)
#   RC_RUN_ID   run to publish from; checked, not trusted (optional)

set -euo pipefail

REPO="${REPO:?REPO is required}"
TAG="${TAG:?TAG is required}"
SHA="${SHA:?SHA is required}"
RC_RUN_ID="${RC_RUN_ID:-}"

if [ -n "${RC_RUN_ID}" ]; then
  run=$(gh run view "${RC_RUN_ID}" --repo "${REPO}" --json headSha,conclusion)
  built=$(jq -r '.headSha' <<< "${run}")
  conclusion=$(jq -r '.conclusion' <<< "${run}")
  if [ "${built}" != "${SHA}" ]; then
    echo "run ${RC_RUN_ID} built ${built}, but ${TAG} points at ${SHA}; refusing to publish a different commit" >&2
    exit 1
  fi
  if [ "${conclusion}" = "cancelled" ]; then
    echo "run ${RC_RUN_ID} was cancelled; refusing to publish from it" >&2
    exit 1
  fi
  echo "${RC_RUN_ID}"
  exit 0
fi

runs=$(gh run list --repo "${REPO}" --workflow=release-candidate.yml \
  --branch "${TAG}" --event push --limit 50 \
  --json databaseId,createdAt,conclusion,headSha)
id=$(jq -r --arg sha "${SHA}" '
  map(select(.headSha == $sha and .conclusion != "cancelled"))
  | sort_by(.createdAt) | last | .databaseId // empty' <<< "${runs}")
if [ -z "${id}" ]; then
  echo "no uncancelled 'Release candidate' run built ${TAG} at ${SHA}; push the tag first, or pass rc_run_id" >&2
  exit 1
fi
echo "${id}"
