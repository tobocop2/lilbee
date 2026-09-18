#!/usr/bin/env bats
# scripts/release_selfheal.sh against the same stubbed gh the watcher uses.

setup() {
  load helpers
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/release_selfheal.sh"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  mkdir -p "${FIXTURE}/runs" "${FIXTURE}/attempts" "${FIXTURE}/conclusions" "${FIXTURE}/bin"
  : > "${FIXTURE}/actions.log"
  echo "[]" > "${FIXTURE}/empty.json"
  echo false > "${FIXTURE}/prerelease"
  echo 1 > "${FIXTURE}/attempts/900"
  echo success > "${FIXTURE}/conclusions/900"
  rc_jobs "build-binaries (ubuntu):success"
  cp "${BATS_TEST_DIRNAME}/stubs/gh" "${FIXTURE}/bin/gh"
  chmod +x "${FIXTURE}/bin/gh"
}

heal() {
  FIXTURE="${FIXTURE}" GITHUB_STEP_SUMMARY="${FIXTURE}/summary.md" \
  GH_TOKEN=stub REPO=tobocop2/lilbee RUN_ID=900 MAX_ATTEMPTS=2 \
  PATH="${FIXTURE}/bin:${PATH}" bash "${SCRIPT}"
}

summary() { cat "${FIXTURE}/summary.md"; }

@test "a candidate with no failed cells is left alone" {
  run heal
  [ "$status" -eq 0 ]
  [ ! -s "${FIXTURE}/actions.log" ]
  [[ "$output" == *"nothing to heal"* ]]
}

@test "a failed cell on the first attempt is rerun" {
  rc_jobs "build-binaries (ubuntu):failure"
  run heal
  [ "$status" -eq 0 ]
  [[ "$(cat "${FIXTURE}/actions.log")" == *"rerun 900"* ]]
}

@test "a cell cancelled by its own time limit is rerun" {
  rc_jobs "build-binaries (compat-windows):cancelled:timeout"
  run heal
  [ "$status" -eq 0 ]
  [[ "$(cat "${FIXTURE}/actions.log")" == *"rerun 900"* ]]
  run summary
  [[ "$output" == *"- build-binaries (compat-windows)"* ]]
}

@test "a run cancelled on purpose heals nothing" {
  echo cancelled > "${FIXTURE}/conclusions/900"
  rc_jobs "build-binaries (compat-windows):cancelled:user" "build-openapi:cancelled:user"
  run heal
  [ "$status" -eq 0 ]
  [ ! -s "${FIXTURE}/actions.log" ]
  [[ "$output" == *"nothing to heal"* ]]
}

@test "a timed-out cell is rerun even though the run itself concluded cancelled" {
  echo cancelled > "${FIXTURE}/conclusions/900"
  rc_jobs "build-binaries (ubuntu):success" "build-binaries (compat-windows):cancelled:timeout"
  run heal
  [ "$status" -eq 0 ]
  [[ "$(cat "${FIXTURE}/actions.log")" == *"rerun 900"* ]]
  run summary
  [[ "$output" == *"- build-binaries (compat-windows)"* ]]
  [[ "$output" != *"ubuntu"* ]]
}

@test "a cell cancelled for a reason the runner did not record heals nothing" {
  rc_jobs "build-binaries (compat-windows):cancelled"
  run heal
  [ "$status" -eq 0 ]
  [ ! -s "${FIXTURE}/actions.log" ]
  [[ "$output" == *"nothing to heal"* ]]
}

@test "an unreadable annotation list stops the heal instead of reporting a clean run" {
  rc_jobs "build-binaries (compat-windows):cancelled:timeout"
  touch "${FIXTURE}/fail_annotations"
  run heal
  [ "$status" -eq 1 ]
  [ ! -s "${FIXTURE}/actions.log" ]
  [[ "$output" == *"cannot read the annotations"* ]]
  [[ "$output" == *"cannot tell which cells dropped out"* ]]
  [[ "$output" != *"nothing to heal"* ]]
}

@test "a cell stopped without exceeding its limit heals nothing" {
  rc_jobs "build-binaries (compat-windows):cancelled:stopped"
  run heal
  [ "$status" -eq 0 ]
  [ ! -s "${FIXTURE}/actions.log" ]
  [[ "$output" == *"nothing to heal"* ]]
}

@test "a candidate already at the attempt bound is not rerun again" {
  rc_jobs "build-binaries (ubuntu):failure"
  echo 2 > "${FIXTURE}/attempts/900"
  run heal
  [ "$status" -eq 1 ]
  [ ! -s "${FIXTURE}/actions.log" ]
  [[ "$output" == *"reached the bound"* ]]
}
