#!/usr/bin/env bats
# tools/qa/artifact_smoke.sh against a stubbed lilbee entrypoint.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/tools/qa/artifact_smoke.sh"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  mkdir -p "${FIXTURE}/bin" "${FIXTURE}/models"
  touch "${FIXTURE}/models/Qwen3-0.6B-Q8_0.gguf" \
        "${FIXTURE}/models/nomic-embed-text-v1.5.Q4_K_M.gguf"
  cp "${BATS_TEST_DIRNAME}/stubs/lilbee-smoke" "${FIXTURE}/bin/lilbee"
  chmod +x "${FIXTURE}/bin/lilbee"
  CALLS="${FIXTURE}/calls"
  : > "${CALLS}"
}

smoke() {
  LILBEE_EXE="${FIXTURE}/bin/lilbee" MODELS_DIR="${FIXTURE}/models" \
  SMOKE_CALLS="${CALLS}" SMOKE_CRAWL_INDEXED="${SMOKE_CRAWL_INDEXED:-1}" \
  SMOKE_FAIL_LEG="${SMOKE_FAIL_LEG:-}" \
  SKIP_CRAWL="${SKIP_CRAWL:-0}" PATH="${FIXTURE}/bin:${PATH}" \
  bash "${SCRIPT}"
}

# A captured leg as it stood before the fix: the command substitution runs
# under set -e, and its output is printed on the following line.
old_leg() {
  SMOKE_CALLS="${CALLS}" SMOKE_FAIL_LEG=search \
  bash -c '
    set -euxo pipefail
    search_out=$("$1" search "blue quartz resonator")
    echo "${search_out}"
    echo "${search_out}" | grep -qi quartz
  ' _ "${FIXTURE}/bin/lilbee"
}

# The error the CLI printed reaches the log as a line of its own, rather than
# inside the `set -x` trace of the assignment that swallowed it.
printed_as_output() {
  printf '%s\n' "$1" | grep -q '^Error: the search leg'
}

@test "control: the old leg names neither the failing leg nor its exit code" {
  run old_leg
  [ "$status" -ne 0 ]
  [[ "$output" != *"FAIL: search exited"* ]]
  ! printed_as_output "$output"
}

@test "every leg runs and the gate passes" {
  run smoke
  [ "$status" -eq 0 ]
  [[ "$(cat "${CALLS}")" == *"self-check-extras"* ]]
  [[ "$(cat "${CALLS}")" == *"search blue quartz resonator"* ]]
  [[ "$(cat "${CALLS}")" == *"ask What frequency"* ]]
  [[ "$(cat "${CALLS}")" == *"add https://example.com"* ]]
  [[ "$output" == *"ARTIFACT SMOKE PASSED: extras, self-check, ingest, search, ask"* ]]
}

@test "a crawled page that is not searchable fails the gate" {
  SMOKE_CRAWL_INDEXED=0 run smoke
  [ "$status" -eq 1 ]
  [[ "$output" == *"crawled page not searchable"* ]]
  [[ "$output" != *"ARTIFACT SMOKE PASSED"* ]]
}

@test "a crawled page that is not searchable still ran ingest, search and ask" {
  SMOKE_CRAWL_INDEXED=0 run smoke
  [ "$status" -eq 1 ]
  [[ "$(cat "${CALLS}")" == *"search blue quartz resonator"* ]]
  [[ "$(cat "${CALLS}")" == *"ask What frequency"* ]]
}

@test "a failing search leg prints the error the CLI printed, then fails" {
  SMOKE_FAIL_LEG=search run smoke
  [ "$status" -eq 1 ]
  printed_as_output "$output"
  [[ "$output" == *"FAIL: search exited 3"* ]]
}

@test "a failing ask leg prints the error the CLI printed, then fails" {
  SMOKE_FAIL_LEG=ask run smoke
  [ "$status" -eq 1 ]
  [[ "$output" == *"could not reach the engine"* ]]
  [[ "$output" == *"FAIL: ask exited 3"* ]]
}

@test "a failing crawl search leg prints the error the CLI printed, then fails" {
  SMOKE_FAIL_LEG=crawl run smoke
  [ "$status" -eq 1 ]
  [[ "$output" == *"could not reach the engine"* ]]
  [[ "$output" == *"FAIL: crawl search exited 3"* ]]
}

@test "a missing model gguf fails before any leg runs" {
  rm "${FIXTURE}/models/Qwen3-0.6B-Q8_0.gguf"
  run smoke
  [ "$status" -eq 1 ]
  [[ "$output" == *"chat model gguf not found"* ]]
  [ ! -s "${CALLS}" ]
}

@test "SKIP_CRAWL leaves the extras and crawl legs unrun" {
  SKIP_CRAWL=1 run smoke
  [ "$status" -eq 0 ]
  [[ "$(cat "${CALLS}")" != *"self-check-extras"* ]]
  [[ "$(cat "${CALLS}")" != *"add https://example.com"* ]]
  [[ "$output" == *"(extras and crawl skipped)"* ]]
}
