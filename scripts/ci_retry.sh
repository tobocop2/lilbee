#!/usr/bin/env bash
# Run a command until it succeeds, up to a bounded number of attempts, with a
# backoff that grows by the attempt number. For a CI step whose only failure
# mode is a transient external service: a package registry, a release download,
# an API gateway.
#
# The retry is blind, like scripts/release_selfheal.sh: matching a service's
# error text to tell a flake from a defect is a list that goes stale.
#
# It also runs under Git Bash on the Windows runners, so `sleep` is the only
# external command it may use. Everything else is a shell builtin.
#
# Usage: ci_retry.sh <attempts> <command> [args...]

set -euo pipefail

readonly BACKOFF_STEP_S=10

attempts="${1-}"
shift || true

case "${attempts}" in
  '' | *[!0-9]*)
    echo "ci_retry.sh: attempts must be a positive integer, got '${attempts}'" >&2
    exit 2
    ;;
esac
if [ "${attempts}" -lt 1 ]; then
  echo "ci_retry.sh: attempts must be a positive integer, got '${attempts}'" >&2
  exit 2
fi
if [ "$#" -eq 0 ]; then
  echo "ci_retry.sh: no command given" >&2
  exit 2
fi

attempt=1
while [ "${attempt}" -le "${attempts}" ]; do
  if "$@"; then
    exit 0
  fi
  if [ "${attempt}" -eq "${attempts}" ]; then
    break
  fi
  delay=$((attempt * BACKOFF_STEP_S))
  echo "ci_retry: attempt ${attempt}/${attempts} of '$1' failed; retrying in ${delay}s" >&2
  sleep "${delay}"
  attempt=$((attempt + 1))
done

echo "ci_retry: '$1' failed after ${attempts} attempts" >&2
exit 1
