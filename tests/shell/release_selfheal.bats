#!/usr/bin/env bats
# scripts/release_selfheal.sh against the same stubbed gh the watcher uses.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/release_selfheal.sh"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  mkdir -p "${FIXTURE}/runs" "${FIXTURE}/attempts" "${FIXTURE}/conclusions" "${FIXTURE}/bin"
  : > "${FIXTURE}/actions.log"
  echo "[]" > "${FIXTURE}/empty.json"
  echo false > "${FIXTURE}/prerelease"
  echo 1 > "${FIXTURE}/attempts/900"
  echo success > "${FIXTURE}/conclusions/900"
  : > "${FIXTURE}/rc_failed"          # job-name lines, not a count
  cp "${BATS_TEST_DIRNAME}/stubs/gh" "${FIXTURE}/bin/gh"
  chmod +x "${FIXTURE}/bin/gh"
}

heal() {
  FIXTURE="${FIXTURE}" GITHUB_STEP_SUMMARY="${FIXTURE}/summary.md" \
  GH_TOKEN=stub REPO=tobocop2/lilbee RUN_ID=900 MAX_ATTEMPTS=2 \
  PATH="${FIXTURE}/bin:${PATH}" bash "${SCRIPT}"
}

@test "a candidate with no failed cells is left alone" {
  run heal
  [ "$status" -eq 0 ]
  [ ! -s "${FIXTURE}/actions.log" ]
  [[ "$output" == *"nothing to heal"* ]]
}

@test "a failed cell on the first attempt is rerun" {
  printf 'build-binaries (ubuntu)\n' > "${FIXTURE}/rc_failed"
  run heal
  [ "$status" -eq 0 ]
  [[ "$(cat "${FIXTURE}/actions.log")" == *"rerun 900"* ]]
}

@test "a candidate already at the attempt bound is not rerun again" {
  printf 'build-binaries (ubuntu)\n' > "${FIXTURE}/rc_failed"
  echo 2 > "${FIXTURE}/attempts/900"
  run heal
  [ "$status" -eq 1 ]
  [ ! -s "${FIXTURE}/actions.log" ]
  [[ "$output" == *"reached the bound"* ]]
}
