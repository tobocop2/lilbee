#!/usr/bin/env bats
# scripts/ci_retry.sh against a command that fails a set number of times.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/ci_retry.sh"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  mkdir -p "${FIXTURE}/bin"
  for stub in flaky sleep; do
    cp "${BATS_TEST_DIRNAME}/stubs/${stub}" "${FIXTURE}/bin/${stub}"
    chmod +x "${FIXTURE}/bin/${stub}"
  done
  COUNTER="${FIXTURE}/calls"
  SLEEP_LOG="${FIXTURE}/sleeps"
  : > "${SLEEP_LOG}"
}

retry() {
  FLAKY_COUNTER="${COUNTER}" FLAKY_FAIL_TIMES="${FLAKY_FAIL_TIMES:-0}" \
  SLEEP_LOG="${SLEEP_LOG}" PATH="${FIXTURE}/bin:${PATH}" \
  bash "${SCRIPT}" "$@"
}

calls() {
  cat "${COUNTER}"
}

@test "a command that succeeds at once runs exactly once" {
  run retry 3 flaky
  [ "$status" -eq 0 ]
  [ "$(calls)" -eq 1 ]
  [ ! -s "${SLEEP_LOG}" ]
}

@test "a command that fails once and then succeeds completes on the second attempt" {
  FLAKY_FAIL_TIMES=1 run retry 3 flaky
  [ "$status" -eq 0 ]
  [ "$(calls)" -eq 2 ]
  [ "$(wc -l < "${SLEEP_LOG}")" -eq 1 ]
  [[ "$output" == *"attempt 1/3"* ]]
}

@test "a command that fails every time stops at the attempt bound" {
  FLAKY_FAIL_TIMES=99 run retry 3 flaky
  [ "$status" -eq 1 ]
  [ "$(calls)" -eq 3 ]
  [[ "$output" == *"failed after 3 attempts"* ]]
}

@test "the bound is honoured at one attempt, so nothing is retried" {
  FLAKY_FAIL_TIMES=99 run retry 1 flaky
  [ "$status" -eq 1 ]
  [ "$(calls)" -eq 1 ]
  [ ! -s "${SLEEP_LOG}" ]
}

@test "the backoff grows with the attempt number" {
  FLAKY_FAIL_TIMES=99 run retry 3 flaky
  [ "$status" -eq 1 ]
  [ "$(sed -n 1p "${SLEEP_LOG}")" -eq 10 ]
  [ "$(sed -n 2p "${SLEEP_LOG}")" -eq 20 ]
}

@test "arguments reach the command unsplit" {
  run retry 2 printf '%s|' "two words" second
  [ "$status" -eq 0 ]
  [[ "$output" == "two words|second|" ]]
}

@test "a non-numeric attempt bound is rejected" {
  run retry lots flaky
  [ "$status" -eq 2 ]
  [ ! -f "${COUNTER}" ]
  [[ "$output" == *"positive integer"* ]]
}

@test "an attempt bound below one is rejected" {
  run retry 0 flaky
  [ "$status" -eq 2 ]
  [ ! -f "${COUNTER}" ]
}

@test "a missing command is rejected" {
  run retry 3
  [ "$status" -eq 2 ]
  [[ "$output" == *"no command"* ]]
}

@test "the retry runs without seq, which Git Bash need not provide" {
  cp "${BATS_TEST_DIRNAME}/stubs/seq" "${FIXTURE}/bin/seq"
  chmod +x "${FIXTURE}/bin/seq"
  FLAKY_FAIL_TIMES=1 run retry 3 flaky
  [ "$status" -eq 0 ]
  [ "$(calls)" -eq 2 ]
  [ "$(wc -l < "${SLEEP_LOG}")" -eq 1 ]
}
