#!/usr/bin/env bats
# scripts/release_watch.sh against a stubbed gh CLI. Every leg is a fixture file
# under $FIXTURE/runs, so a scenario is one jq edit away.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/release_watch.sh"
  TAG="v0.6.90b999"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  mkdir -p "${FIXTURE}/runs" "${FIXTURE}/attempts" "${FIXTURE}/conclusions" "${FIXTURE}/bin"

  : > "${FIXTURE}/actions.log"
  echo "[]" > "${FIXTURE}/empty.json"
  echo false > "${FIXTURE}/prerelease"

  # The release candidate that dispatched the legs: green, nothing to heal.
  echo 1 > "${FIXTURE}/attempts/900"
  echo success > "${FIXTURE}/conclusions/900"
  : > "${FIXTURE}/rc_failed"          # job-name lines; empty means no failed cells

  cp "${BATS_TEST_DIRNAME}/stubs/gh" "${FIXTURE}/bin/gh"
  chmod +x "${FIXTURE}/bin/gh"
  stub_curl 22   # the version is not on PyPI unless a test says otherwise

  # A virtual clock. `date +%s` reads it and `sleep` advances it, so a poll loop
  # runs its passes instantly and deterministically instead of against wall time.
  echo 1000000 > "${FIXTURE}/clock"
  cat > "${FIXTURE}/bin/date" <<'CLOCK'
#!/usr/bin/env bash
[ "${1:-}" = "+%s" ] && { cat "${FIXTURE}/clock"; exit 0; }
exec /bin/date "$@"
CLOCK
  cat > "${FIXTURE}/bin/sleep" <<'CLOCK'
#!/usr/bin/env bash
echo $(( $(cat "${FIXTURE}/clock") + ${1:-0} )) > "${FIXTURE}/clock"
CLOCK
  chmod +x "${FIXTURE}/bin/date" "${FIXTURE}/bin/sleep"

  local i=1
  while IFS='|' read -r file label; do
    make_leg "${file}" "${label}" "$((100 + i))" success
    i=$((i + 1))
  done <<'LEGS'
publish.yml|Publish to PyPI
publish-packages.yml|Publish Packages
publish-cuda-packages.yml|Publish CUDA Packages
publish-rocm-packages.yml|Publish ROCm Packages
publish-compat-packages.yml|Publish compat packages
publish-docker.yml|Publish Docker Image
publish-flatpak.yml|Publish Flatpak
verify-release.yml|Verify release
LEGS
}

make_leg() {  # file  label  run-id  conclusion
  printf '[{"databaseId":%d,"status":"completed","conclusion":"%s","displayTitle":"%s %s"}]\n' \
    "$3" "$4" "$2" "${TAG}" > "${FIXTURE}/runs/$1.json"
  echo 1 > "${FIXTURE}/attempts/$3"
}

set_conclusion() {  # file  conclusion
  jq -c --arg c "$2" 'map(.conclusion = $c)' "${FIXTURE}/runs/$1.json" > "${FIXTURE}/t"
  mv "${FIXTURE}/t" "${FIXTURE}/runs/$1.json"
}

stub_curl() {  # exit-code
  printf '#!/usr/bin/env bash\nexit %s\n' "$1" > "${FIXTURE}/bin/curl"
  chmod +x "${FIXTURE}/bin/curl"
}

watch() {
  FIXTURE="${FIXTURE}" \
  GITHUB_STEP_SUMMARY="${FIXTURE}/summary.md" \
  GH_TOKEN=stub REPO=tobocop2/lilbee TAG="${TAG}" RC_RUN_ID=900 \
  DRY_RUN="${DRY_RUN:-false}" MAX_ATTEMPTS=2 POLL_SECONDS=60 \
  APPEAR_MINUTES="${APPEAR_MINUTES:-0}" WATCH_MINUTES="${WATCH_MINUTES:-5}" \
  PATH="${FIXTURE}/bin:${PATH}" \
    bash "${SCRIPT}"
}

dispatch_log() { cat "${FIXTURE}/actions.log"; }
summary() { cat "${FIXTURE}/summary.md"; }

@test "every leg green and the release promoted: exits 0 and touches nothing" {
  run watch
  [ "$status" -eq 0 ]
  [ ! -s "${FIXTURE}/actions.log" ]
  run summary
  [[ "$output" == *"Every channel published"* ]]
}

@test "a leg that failed on its first attempt is rerun and heals" {
  set_conclusion publish-packages.yml failure
  run watch
  [ "$status" -eq 0 ]
  run dispatch_log
  [[ "$output" == *"rerun 102"* ]]
}

@test "a leg already at the attempt bound is called a defect, not rerun again" {
  set_conclusion publish-flatpak.yml failure
  echo 2 > "${FIXTURE}/attempts/107"
  run watch
  [ "$status" -eq 1 ]
  [ ! -s "${FIXTURE}/actions.log" ]
  run summary
  [[ "$output" == *"failed on attempt 2"* ]]
}

@test "a rerun that fails again stops at the bound" {
  set_conclusion publish-docker.yml failure
  touch "${FIXTURE}/no_heal_106"
  run watch
  [ "$status" -eq 1 ]
  [ "$(grep -c 'rerun 106' "${FIXTURE}/actions.log")" -eq 1 ]
}

@test "a leg whose dispatch was lost is dispatched again exactly once" {
  rm "${FIXTURE}/runs/publish-docker.yml.json"
  run watch
  [ "$status" -eq 1 ]
  [ "$(grep -c 'dispatch publish-docker.yml' "${FIXTURE}/actions.log")" -eq 1 ]
}

@test "a re-dispatched leg that lands is green" {
  cp "${FIXTURE}/runs/publish-docker.yml.json" "${FIXTURE}/appear_on_dispatch"
  rm "${FIXTURE}/runs/publish-docker.yml.json"
  run watch
  [ "$status" -eq 0 ]
}

@test "verify-release is never re-dispatched, because a dispatched copy cannot promote" {
  rm "${FIXTURE}/runs/verify-release.yml.json"
  run watch
  [ "$status" -eq 1 ]
  run dispatch_log
  [[ "$output" != *"verify-release.yml"* ]]
}

@test "while the candidate is still healing the watcher defers and changes nothing" {
  printf 'cell-a\ncell-b\ncell-c\n' > "${FIXTURE}/rc_failed"
  set_conclusion publish-packages.yml failure
  run watch
  [ "$status" -eq 0 ]
  [ ! -s "${FIXTURE}/actions.log" ]
  run summary
  [[ "$output" == *"release-selfheal owns this"* ]]
}

@test "a candidate already at the attempt bound is watched rather than deferred to" {
  echo 2 > "${FIXTURE}/attempts/900"
  printf 'cell-a\n' > "${FIXTURE}/rc_failed"
  run watch
  [ "$status" -eq 0 ]
}

@test "a red publish.yml whose version is live on PyPI spends no retry" {
  set_conclusion publish.yml failure
  stub_curl 0
  run watch
  [ "$status" -eq 0 ]
  [ ! -s "${FIXTURE}/actions.log" ]
  run summary
  [[ "$output" == *"is live on PyPI"* ]]
}

@test "every leg green but an unpromoted release is still a failure" {
  echo true > "${FIXTURE}/prerelease"
  run watch
  [ "$status" -eq 1 ]
  run summary
  [[ "$output" == *"is not promoted"* ]]
}

@test "a cancelled leg is reported, not rerun" {
  set_conclusion publish-cuda-packages.yml cancelled
  run watch
  [ "$status" -eq 1 ]
  [ ! -s "${FIXTURE}/actions.log" ]
  run summary
  [[ "$output" == *"concluded cancelled"* ]]
}

@test "a leg still running when the budget runs out is reported unsettled" {
  jq -c 'map(.status = "in_progress")' "${FIXTURE}/runs/publish-packages.yml.json" > "${FIXTURE}/t"
  mv "${FIXTURE}/t" "${FIXTURE}/runs/publish-packages.yml.json"
  run watch
  [ "$status" -eq 1 ]
  run summary
  [[ "$output" == *"unsettled"* ]]
}

@test "dry run reports what it would do and issues nothing" {
  set_conclusion publish-packages.yml failure
  DRY_RUN=true run watch
  [ "$status" -eq 1 ]
  [ ! -s "${FIXTURE}/actions.log" ]
  run summary
  [[ "$output" == *"would rerun"* ]]
}

@test "two legs whose dispatch was lost are re-dispatched in the same pass" {
  # A shared appear deadline made one re-dispatch push every other missing leg
  # out by another APPEAR_MINUTES, so seven lost dispatches healed in series.
  rm "${FIXTURE}/runs/publish-docker.yml.json" "${FIXTURE}/runs/publish-cuda-packages.yml.json"
  APPEAR_MINUTES=20 WATCH_MINUTES=60 run watch
  [ "$status" -eq 1 ]
  [ "$(grep -c 'dispatch publish-docker.yml' "${FIXTURE}/actions.log")" -eq 1 ]
  [ "$(grep -c 'dispatch publish-cuda-packages.yml' "${FIXTURE}/actions.log")" -eq 1 ]
}

@test "a failing run list leaves the leg pending and dispatches nothing" {
  # An API error must not read as "this leg never ran": that path issues a real
  # dispatch and would publish a duplicate.
  rm "${FIXTURE}/runs/publish-docker.yml.json"
  touch "${FIXTURE}/fail_run_list"
  run watch
  [ "$status" -eq 1 ]
  [ ! -s "${FIXTURE}/actions.log" ]
  run summary
  [[ "$output" == *"unsettled"* ]]
}

@test "an unreadable candidate is watched, not mistaken for one that is healing" {
  # Deferring on an API error would end the watch having watched nothing, and
  # self-heal only fires on a candidate that actually failed, so nothing would
  # follow. Every leg here is green, so watching correctly ends green too.
  touch "${FIXTURE}/fail_api"
  run watch
  [ "$status" -eq 0 ]
  [[ "$output" == *"could not read candidate"* ]]
  run summary
  [[ "$output" != *"release-selfheal owns this"* ]]
  [[ "$output" == *"Every channel published"* ]]
}
