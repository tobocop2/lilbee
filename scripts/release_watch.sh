#!/usr/bin/env bash
# Watch the workflows a release candidate dispatches, and heal the ones that flake.
#
# release-selfheal.yml reruns the candidate's own build cells. Nothing watched
# the seven workflows the candidate dispatches, nor verify-release which fires
# on its completion, so a flake in a publish leg (an
# apt 403 on a runner image, an AUR maintenance window, a nix job racing a push
# to main) left that channel unpublished until a person noticed and pressed
# rerun. This script is that person.
#
# It waits for all eight legs, reruns a failed one under the same attempt
# bound release-selfheal uses, re-issues a dispatch that never landed, and exits
# non-zero when a leg burns both attempts.
#
# Run it by hand against a finished release:
#   TAG=v0.6.90b434 DRY_RUN=true bash scripts/release_watch.sh
#
# Environment:
#   TAG             release tag to watch (required)
#   REPO            owner/name (default: tobocop2/lilbee)
#   RC_RUN_ID       release-candidate run that dispatched the legs (optional;
#                   when set, the script defers while that run is still healing)
#   DRY_RUN         true reports what it would do and changes nothing
#   MAX_ATTEMPTS    attempts per leg before it is called a defect (default 2)
#   APPEAR_MINUTES  how long a leg may take to show up before its dispatch is
#                   assumed lost and issued again (default 20)
#   WATCH_MINUTES   total watch budget (default 290)
#   PACKAGE         PyPI project name for the already-published check (default lilbee)

set -uo pipefail

TAG="${TAG:?TAG is required, e.g. TAG=v0.6.90b434}"
REPO="${REPO:-tobocop2/lilbee}"
RC_RUN_ID="${RC_RUN_ID:-}"
DRY_RUN="${DRY_RUN:-false}"
# One retry, matching release-selfheal.yml. A second failure on the same leg is
# a defect until proven otherwise, and the bound is the only thing between a
# real defect and an unbounded rerun loop.
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
APPEAR_MINUTES="${APPEAR_MINUTES:-20}"
WATCH_MINUTES="${WATCH_MINUTES:-290}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PACKAGE="${PACKAGE:-lilbee}"
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/stdout}"

# file | run-name prefix | may this leg be re-dispatched?
#
# verify-release must not be: its promote job is gated on
# github.event_name == 'workflow_run', so a dispatched copy verifies the tag but
# never marks it latest.
LEGS='publish.yml|Publish to PyPI|yes
publish-packages.yml|Publish Packages|yes
publish-cuda-packages.yml|Publish CUDA Packages|yes
publish-rocm-packages.yml|Publish ROCm Packages|yes
publish-compat-packages.yml|Publish compat packages|yes
publish-docker.yml|Publish Docker Image|yes
publish-flatpak.yml|Publish Flatpak|yes
verify-release.yml|Verify release|no'

state_dir="$(mktemp -d "${TMPDIR:-/tmp}/release-watch.XXXXXX")"
trap 'rm -rf "${state_dir}"' EXIT

note() { echo "$*"; }
now() { date +%s; }

settle() {  # leg-file  state  reason
  echo "$2" > "${state_dir}/$1.state"
  echo "$3" > "${state_dir}/$1.reason"
  note "${1}: ${2} (${3})"
}

# Defer to release-selfheal while the candidate is still healing.
#
# A soft build cell that drops its artifact leaves the candidate GREEN with an
# asset missing. verify-release fires on the same completion and fails on the
# gap, while release-selfheal reruns the dropped cell. Rerunning verify-release
# now would fail it again on the same missing asset and spend both its attempts
# before the healed cell re-attaches anything. Self-heal's rerun emits a fresh
# candidate completion, which starts a fresh watch, so nothing is lost by
# leaving.
defer_to_selfheal() {
  [ -n "${RC_RUN_ID}" ] || return 1

  local rc rc_attempt rc_conclusion rc_failed
  # An unreadable candidate is not evidence that it is healing. Deferring on an
  # API error would end the watch having watched nothing, and self-heal only
  # fires on a candidate that actually failed, so nothing would follow.
  rc=$(gh api "repos/${REPO}/actions/runs/${RC_RUN_ID}" \
         -q '"\(.run_attempt) \(.conclusion)"') || {
    note "could not read candidate ${RC_RUN_ID}; watching the legs rather than assuming a heal"
    return 1
  }
  rc_attempt="${rc%% *}"
  rc_conclusion="${rc#* }"
  # No --paginate: the jq runs per page, so a second page makes this "0\n0" and
  # the numeric test below errors instead of comparing. per_page=100 covers the
  # candidate's job count, and release-selfheal.yml queries the same way.
  rc_failed=$(gh api "repos/${REPO}/actions/runs/${RC_RUN_ID}/jobs?per_page=100" \
                -q '[.jobs[] | select(.conclusion == "failure")] | length') || {
    note "could not read candidate ${RC_RUN_ID} jobs; watching the legs"
    return 1
  }

  if [ "${rc_conclusion}" = "success" ] && [ "${rc_failed:-0}" -eq 0 ]; then
    return 1
  fi
  if [ "${rc_attempt:-1}" -lt "${MAX_ATTEMPTS}" ]; then
    note "candidate ${RC_RUN_ID} concluded ${rc_conclusion} with ${rc_failed} failed cell(s) on attempt ${rc_attempt}."
    note "release-selfheal owns this; its rerun starts a fresh watch. Nothing to do."
    {
      echo "## Release watch ${TAG}"
      echo
      echo "release-selfheal owns this candidate: it concluded ${rc_conclusion} with ${rc_failed} failed cell(s) on attempt ${rc_attempt}."
    } >> "${SUMMARY}"
    return 0
  fi
  note "candidate ${RC_RUN_ID} still has ${rc_failed} failed cell(s) at attempt ${rc_attempt}; watching the legs that did dispatch."
  return 1
}

find_run() {  # workflow-file  run-title -> the newest matching run as JSON, or empty
  # Exit status matters: an API error must not read as "this leg never ran",
  # because that path issues a real dispatch and would publish a duplicate.
  # --arg rather than string interpolation so the title cannot alter the filter.
  gh run list --workflow="$1" --repo "${REPO}" --limit 30 \
    --json databaseId,status,conclusion,displayTitle \
    | jq -c --arg title "$2" '[.[] | select(.displayTitle == $title)] | first'
}

# publish.yml's guard refuses when the version is already live on PyPI, which is
# the documented benign failure. The goal is the wheel being on PyPI, not the run
# being green, so check the goal before spending a retry on it.
version_is_on_pypi() {
  curl -fsS -o /dev/null "https://pypi.org/pypi/${PACKAGE}/${TAG#v}/json"
}

handle_missing_leg() {  # file  title  redispatchable
  local file="$1" title="$2" redispatchable="$3" deadline_file="${state_dir}/$1.deadline"

  # Per leg, not shared. A shared deadline meant one re-dispatch pushed every
  # other missing leg out by another APPEAR_MINUTES, so healing seven lost
  # dispatches took seven times the wait, in series, for no reason.
  [ -f "${deadline_file}" ] || echo "$(( $(now) + APPEAR_MINUTES * 60 ))" > "${deadline_file}"
  [ "$(now)" -lt "$(cat "${deadline_file}")" ] && return 0

  if [ "${redispatchable}" != "yes" ] || [ -f "${state_dir}/${file}.redispatched" ]; then
    settle "${file}" red "never appeared"
    return 0
  fi

  touch "${state_dir}/${file}.redispatched"
  if [ "${DRY_RUN}" = "true" ]; then
    note "${file}: would re-dispatch (no run titled '${title}' after ${APPEAR_MINUTES}m)"
    settle "${file}" red "dry run: never appeared"
    return 0
  fi
  note "${file}: no run titled '${title}' after ${APPEAR_MINUTES}m; the dispatch was lost, issuing it again"
  gh workflow run "${file}" --repo "${REPO}" -f tag="${TAG}" || true
  echo "$(( $(now) + APPEAR_MINUTES * 60 ))" > "${deadline_file}"
}

handle_completed_leg() {  # file  run-json
  local file="$1" run="$2" conclusion run_id attempt
  conclusion=$(echo "${run}" | jq -r .conclusion)
  run_id=$(echo "${run}" | jq -r .databaseId)

  if [ "${conclusion}" = "success" ]; then
    settle "${file}" green "run ${run_id}"
    return 0
  fi
  if [ "${file}" = "publish.yml" ] && version_is_on_pypi; then
    settle "${file}" green "${PACKAGE} ${TAG#v} is live on PyPI"
    return 0
  fi
  if [ "${conclusion}" != "failure" ]; then
    settle "${file}" red "run ${run_id} concluded ${conclusion}"
    return 0
  fi

  attempt=$(gh api "repos/${REPO}/actions/runs/${run_id}" -q .run_attempt 2>/dev/null) || attempt="${MAX_ATTEMPTS}"
  if [ "${attempt}" -ge "${MAX_ATTEMPTS}" ]; then
    settle "${file}" red "run ${run_id} failed on attempt ${attempt}"
    return 0
  fi
  if [ "${DRY_RUN}" = "true" ]; then
    settle "${file}" red "dry run: would rerun ${run_id} (attempt ${attempt})"
    return 0
  fi
  note "${file}: run ${run_id} failed on attempt ${attempt}; rerunning its failed jobs"
  gh run rerun "${run_id}" --repo "${REPO}" --failed || true
}

# `gh run watch <id> --exit-status` owns "block on one run, exit non-zero if it
# failed", and if this ever waited on a single known run it should call that.
# It cannot own this: a leg that never appeared has no run id to watch, and the
# fan-in needs one deadline and one attempt bound across all eight rather than
# eight blocking calls and a wait.
watch_legs() {
  local watch_deadline pending file label redispatchable title run status
  watch_deadline=$(( $(now) + WATCH_MINUTES * 60 ))

  while true; do
    pending=0
    while IFS='|' read -r file label redispatchable; do
      [ -n "${file}" ] || continue
      [ -f "${state_dir}/${file}.state" ] && continue
      pending=1
      title="${label} ${TAG}"

      if ! run=$(find_run "${file}" "${title}"); then
        note "${file}: could not list runs; leaving it pending rather than re-dispatching"
        continue
      fi
      if [ -z "${run}" ] || [ "${run}" = "null" ]; then
        handle_missing_leg "${file}" "${title}" "${redispatchable}"
        continue
      fi

      status=$(echo "${run}" | jq -r .status)
      [ "${status}" = "completed" ] || continue
      handle_completed_leg "${file}" "${run}"
    done <<< "${LEGS}"

    [ "${pending}" -eq 0 ] && return 0
    if [ "$(now)" -ge "${watch_deadline}" ]; then
      note "watch budget of ${WATCH_MINUTES}m is spent; the legs below are still unsettled"
      return 0
    fi
    sleep "${POLL_SECONDS}"
  done
}

report() {  # -> 0 when everything shipped
  local failed=0 file label redispatchable state reason prerelease
  {
    echo "## Release watch ${TAG}"
    echo
    echo "| Leg | State | Detail |"
    echo "| --- | --- | --- |"
  } >> "${SUMMARY}"

  while IFS='|' read -r file label redispatchable; do
    [ -n "${file}" ] || continue
    if [ -f "${state_dir}/${file}.state" ]; then
      state=$(cat "${state_dir}/${file}.state")
      reason=$(cat "${state_dir}/${file}.reason")
    else
      state="unsettled"
      reason="still running when the watch budget ran out"
    fi
    [ "${state}" = "green" ] || failed=1
    echo "| ${file} | ${state} | ${reason} |" >> "${SUMMARY}"
  done <<< "${LEGS}"

  # verify-release going green means its promote job marked the release latest.
  # Asserting it directly costs one call and states the actual definition of a
  # shipped release rather than inferring it.
  prerelease=$(gh release view "${TAG}" --repo "${REPO}" --json isPrerelease -q .isPrerelease 2>/dev/null) || prerelease="unknown"
  if [ "${prerelease}" = "false" ]; then
    echo "| (release) | green | ${TAG} is promoted to latest |" >> "${SUMMARY}"
  else
    failed=1
    echo "| (release) | red | ${TAG} is not promoted (isPrerelease=${prerelease}) |" >> "${SUMMARY}"
  fi

  echo >> "${SUMMARY}"
  if [ "${failed}" -eq 0 ]; then
    echo "Every channel published and ${TAG} is latest. Nothing was left for a person." >> "${SUMMARY}"
    return 0
  fi
  echo "A leg above did not reach green. Each failing leg was retried once; read its logs before rerunning, because a second failure is a defect until proven otherwise." >> "${SUMMARY}"
  return 1
}

main() {
  if defer_to_selfheal; then
    return 0
  fi
  watch_legs
  report
}

main "$@"
