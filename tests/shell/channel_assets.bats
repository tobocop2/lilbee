#!/usr/bin/env bats
# scripts/channel_asset_blockers.sh against the stubbed gh, and its call site.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/channel_asset_blockers.sh"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  mkdir -p "${FIXTURE}/runs" "${FIXTURE}/views" "${FIXTURE}/bin"
  : > "${FIXTURE}/actions.log"
  : > "${FIXTURE}/release_assets"
  echo "[]" > "${FIXTURE}/empty.json"
  cp "${BATS_TEST_DIRNAME}/stubs/gh" "${FIXTURE}/bin/gh"
  chmod +x "${FIXTURE}/bin/gh"
}

blockers() {  # asset... ; RC_RUN_ID comes from the caller
  FIXTURE="${FIXTURE}" GH_TOKEN=stub REPO=tobocop2/lilbee TAG=v1.0.0 \
  PATH="${FIXTURE}/bin:${PATH}" bash "${SCRIPT}" "$@"
}

publisher_run() {  # workflow-file run-name-prefix status conclusion
  printf '[{"databaseId":55,"status":"%s","conclusion":"%s","displayTitle":"%s v1.0.0"}]\n' \
    "$3" "$4" "$2" > "${FIXTURE}/runs/$1.json"
}

rc_dispatch() {  # job:conclusion ...
  local jobs="" entry
  for entry in "$@"; do
    jobs="${jobs}${jobs:+,}{\"name\":\"${entry%%:*}\",\"conclusion\":\"${entry#*:}\"}"
  done
  printf '{"jobs":[%s]}\n' "${jobs}" > "${FIXTURE}/views/900.json"
}

@test "a skipped dispatch job blocks the artifact it would have produced" {
  rc_dispatch "dispatch-rocm:skipped" "dispatch-cuda:success"
  RC_RUN_ID=900 run blockers lilbee-linux-x86_64-rocm.snap
  [ "$status" -eq 0 ]
  [[ "$output" == "lilbee-linux-x86_64-rocm.snap: dispatch-rocm skipped"* ]]
}

@test "a failed dispatch job blocks its artifact" {
  rc_dispatch "dispatch-compat:failure"
  RC_RUN_ID=900 run blockers lilbee-compat-linux-x86_64.snap
  [[ "$output" == *"dispatch-compat failure"* ]]
}

@test "a dispatch job that succeeded and has no run yet is still awaited" {
  rc_dispatch "dispatch-rocm:success"
  RC_RUN_ID=900 run blockers lilbee-linux-x86_64-rocm.snap
  [ "$status" -eq 0 ]
  [ -z "$output" ]
}

@test "a publisher run that finished without the artifact blocks it" {
  publisher_run publish-flatpak.yml "Publish Flatpak" completed failure
  run blockers lilbee-cuda.flatpakref
  [ "$status" -eq 0 ]
  [[ "$output" == *"publish-flatpak.yml run 55 finished (failure) without it"* ]]
}

@test "a publisher run that finished green without the artifact blocks it too" {
  publisher_run publish-packages.yml "Publish Packages" completed success
  run blockers lilbee-linux-x86_64.snap
  [[ "$output" == *"finished (success) without it"* ]]
}

@test "a publisher run still in progress is awaited" {
  publisher_run publish-flatpak.yml "Publish Flatpak" in_progress ""
  rc_dispatch "dispatch-fanout:success"
  RC_RUN_ID=900 run blockers lilbee.flatpakref
  [ "$status" -eq 0 ]
  [ -z "$output" ]
}

@test "a publisher run for another tag does not count" {
  printf '[{"databaseId":55,"status":"completed","conclusion":"failure","displayTitle":"Publish Flatpak v0.9.9"}]\n' \
    > "${FIXTURE}/runs/publish-flatpak.yml.json"
  run blockers lilbee.flatpakref
  [ "$status" -eq 0 ]
  [ -z "$output" ]
}

@test "an artifact that landed while the verdicts were read is not blocked" {
  publisher_run publish-flatpak.yml "Publish Flatpak" completed success
  echo lilbee.flatpakref > "${FIXTURE}/release_assets"
  run blockers lilbee.flatpakref
  [ "$status" -eq 0 ]
  [ -z "$output" ]
}

@test "only the missing artifact of two is reported" {
  rc_dispatch "dispatch-rocm:skipped" "dispatch-cuda:skipped"
  RC_RUN_ID=900 run blockers lilbee-linux-x86_64-rocm.snap lilbee-linux-x86_64-cu125.snap
  [ "$(echo "${output}" | wc -l | tr -d ' ')" -eq 2 ]
}

@test "an unreadable release candidate is a failed lookup, not a verdict" {
  RC_RUN_ID=901 run blockers lilbee-linux-x86_64-rocm.snap
  [ "$status" -ne 0 ]
}

# Runs the real wait step of verify-release.yml, extracted with yq, from the
# repository root. sleep is stubbed, so a wait that polls again is visible.
wait_step() {
  yq -r '.jobs["verify-channel-assets"].steps[1].run' "${REPO_ROOT}/.github/workflows/verify-release.yml" \
    > "${BATS_TEST_TMPDIR}/wait.sh"
  # A wait that never ends would hang the suite: the third poll stops the step.
  cat > "${FIXTURE}/bin/sleep" <<'STUB'
#!/usr/bin/env bash
echo "$1" >> "${SLEEP_LOG:?SLEEP_LOG is required}"
[ "$(wc -l < "${SLEEP_LOG}")" -lt 3 ] || kill -9 "${PPID}"
STUB
  chmod +x "${FIXTURE}/bin/sleep"
  cd "${REPO_ROOT}"
  FIXTURE="${FIXTURE}" SLEEP_LOG="${FIXTURE}/sleep.log" GH_TOKEN=stub \
  GITHUB_REPOSITORY=tobocop2/lilbee REPO=tobocop2/lilbee TAG=v1.0.0 \
  RC_RUN_ID="${RC_RUN_ID:-900}" WAIT_MINUTES="${WAIT_MINUTES:-180}" \
  PATH="${FIXTURE}/bin:${PATH}" bash "${BATS_TEST_TMPDIR}/wait.sh"
}

@test "the wait step fails at once when the rocm dispatch was skipped" {
  : > "${FIXTURE}/sleep.log"
  rc_dispatch "dispatch-fanout:success" "dispatch-cuda:success" "dispatch-rocm:skipped"
  run wait_step
  [ "$status" -eq 1 ]
  [[ "$output" == *"channel artifacts that cannot arrive"* ]]
  [[ "$output" == *"lilbee-linux-x86_64-rocm.snap: dispatch-rocm skipped"* ]]
  [ ! -s "${FIXTURE}/sleep.log" ]
}

@test "the wait step keeps polling while every publisher can still deliver" {
  : > "${FIXTURE}/sleep.log"
  rc_dispatch "dispatch-fanout:success" "dispatch-cuda:success" "dispatch-rocm:success"
  WAIT_MINUTES=0 run wait_step
  [ "$status" -eq 1 ]
  [[ "$output" == *"still missing after 0m"* ]]
  [[ "$output" != *"cannot arrive"* ]]
}
