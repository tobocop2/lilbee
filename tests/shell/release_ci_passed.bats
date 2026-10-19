#!/usr/bin/env bats
# scripts/release_ci_passed.sh: a release skips its own lint and test only when
# CI tested this exact commit, not when CI passed by skipping every job.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/release_ci_passed.sh"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  mkdir -p "${FIXTURE}/bin" "${FIXTURE}/jobs"
  cp "${BATS_TEST_DIRNAME}/stubs/gh-ci-runs" "${FIXTURE}/bin/gh"
  chmod +x "${FIXTURE}/bin/gh"
  echo abc123 > "${FIXTURE}/sha"
  echo '{"workflow_runs":[]}' > "${FIXTURE}/runs.json"
}

# runs id:conclusion ...
runs() {
  local entry out=""
  for entry in "$@"; do
    out="${out}${out:+,}{\"id\":${entry%%:*},\"conclusion\":\"${entry#*:}\"}"
  done
  printf '{"workflow_runs":[%s]}\n' "${out}" > "${FIXTURE}/runs.json"
}

# jobs run_id name:conclusion ...
jobs() {
  local id="$1" entry out=""
  shift
  for entry in "$@"; do
    out="${out}${out:+,}{\"name\":\"${entry%%:*}\",\"conclusion\":\"${entry#*:}\"}"
  done
  printf '{"jobs":[%s]}\n' "${out}" > "${FIXTURE}/jobs/${id}.json"
}

passed() {
  FIXTURE="${FIXTURE}" PATH="${FIXTURE}/bin:${PATH}" bash "${SCRIPT}" tobocop2/lilbee v1.2.3
}

@test "a successful run that linted and tested the commit passes" {
  runs 7:success
  jobs 7 changes:success lint:success "test (ubuntu-latest, 3.12)":success ci-result:success
  run passed
  [ "$status" -eq 0 ]
  [ "${lines[-1]}" = "ci_passed=true" ]
}

@test "a successful run that skipped every job does not pass" {
  runs 7:success
  jobs 7 changes:success lint:skipped "test":skipped ci-result:success
  run passed
  [ "$status" -eq 0 ]
  [ "${lines[-1]}" = "ci_passed=false" ]
}

@test "a skipped run beside a tested one still passes" {
  runs 7:success 8:success
  jobs 7 changes:success lint:skipped ci-result:success
  jobs 8 changes:success lint:success ci-result:success
  run passed
  [ "${lines[-1]}" = "ci_passed=true" ]
}

@test "a failed run does not pass" {
  runs 7:failure
  jobs 7 lint:success "test (ubuntu-latest, 3.12)":failure
  run passed
  [ "${lines[-1]}" = "ci_passed=false" ]
}

@test "no run for the commit does not pass" {
  run passed
  [ "$status" -eq 0 ]
  [ "${lines[-1]}" = "ci_passed=false" ]
}

@test "a ref that does not resolve does not pass" {
  rm "${FIXTURE}/sha"
  run passed
  [ "$status" -eq 0 ]
  [ "${lines[-1]}" = "ci_passed=false" ]
}

@test "a job list that cannot be read does not pass" {
  runs 7:success
  run passed
  [ "$status" -eq 0 ]
  [ "${lines[-1]}" = "ci_passed=false" ]
}

@test "the verdict is the only line on stdout, the step output it writes" {
  runs 7:success
  jobs 7 lint:success
  out=$(passed 2>/dev/null)
  [ "${out}" = "ci_passed=true" ]
}

@test "the release gate asks this script" {
  run grep -c 'bash scripts/release_ci_passed.sh' "${REPO_ROOT}/.github/workflows/release.yml"
  [ "$output" = "1" ]
}
