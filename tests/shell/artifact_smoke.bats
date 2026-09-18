#!/usr/bin/env bats
# tools/qa/artifact_smoke.sh against a stubbed entrypoint and a stubbed probe.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/tools/qa/artifact_smoke.sh"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  mkdir -p "${FIXTURE}/bin" "${FIXTURE}/models"
  touch "${FIXTURE}/models/Qwen3-0.6B-Q8_0.gguf" \
        "${FIXTURE}/models/nomic-embed-text-v1.5.Q4_K_M.gguf"
  cp "${BATS_TEST_DIRNAME}/stubs/lilbee-smoke" "${FIXTURE}/bin/lilbee"
  cp "${BATS_TEST_DIRNAME}/stubs/curl" "${FIXTURE}/bin/curl"
  chmod +x "${FIXTURE}/bin/lilbee" "${FIXTURE}/bin/curl"
  CALLS="${FIXTURE}/calls"
  CURL_LOG="${FIXTURE}/curl.log"
  : > "${CALLS}"
  : > "${CURL_LOG}"
}

smoke() {
  LILBEE_EXE="${FIXTURE}/bin/lilbee" MODELS_DIR="${FIXTURE}/models" \
  SMOKE_CALLS="${CALLS}" SMOKE_CRAWL_INDEXED="${SMOKE_CRAWL_INDEXED:-1}" \
  CURL_LOG="${CURL_LOG}" CURL_STUB_RC="${CURL_STUB_RC:-0}" \
  SKIP_CRAWL="${SKIP_CRAWL:-0}" PATH="${FIXTURE}/bin:${PATH}" \
  bash "${SCRIPT}"
}

@test "a reachable crawl host runs the crawl leg" {
  run smoke
  [ "$status" -eq 0 ]
  [[ "$(cat "${CALLS}")" == *"add https://example.com"* ]]
  [[ "$output" == *"ARTIFACT SMOKE PASSED: extras, self-check, ingest, search, ask"* ]]
  [[ "$output" != *"is unreachable from this runner"* ]]
}

@test "an unreachable crawl host fails the leg" {
  CURL_STUB_RC=6 run smoke
  [ "$status" -eq 1 ]
  [[ "$(cat "${CALLS}")" != *"add https://example.com"* ]]
  [[ "$output" == *"FAIL: https://example.com is unreachable from this runner (curl rc 6)"* ]]
  [[ "$output" != *"ARTIFACT SMOKE PASSED"* ]]
}

@test "an unreachable crawl host still ran the extras leg first" {
  CURL_STUB_RC=6 run smoke
  [ "$status" -eq 1 ]
  [[ "$(cat "${CALLS}")" == *"self-check-extras"* ]]
}

@test "an unreachable crawl host still ran ingest, search and ask first" {
  CURL_STUB_RC=6 run smoke
  [ "$status" -eq 1 ]
  [[ "$(cat "${CALLS}")" == *"search blue quartz resonator"* ]]
  [[ "$(cat "${CALLS}")" == *"ask What frequency"* ]]
}

@test "a missing probe tool fails instead of skipping the crawl leg" {
  CURL_STUB_RC=127 run smoke
  [ "$status" -eq 1 ]
  [[ "$output" == *"curl is required to probe"* ]]
  [[ "$output" != *"ARTIFACT SMOKE PASSED"* ]]
}

@test "a reachable host whose crawled page is unsearchable still fails" {
  SMOKE_CRAWL_INDEXED=0 run smoke
  [ "$status" -eq 1 ]
  [[ "$output" == *"crawled page not searchable"* ]]
}

@test "SKIP_CRAWL leaves the extras and crawl legs unrun and probes nothing" {
  SKIP_CRAWL=1 run smoke
  [ "$status" -eq 0 ]
  [[ "$(cat "${CALLS}")" != *"self-check-extras"* ]]
  [[ "$(cat "${CALLS}")" != *"add https://example.com"* ]]
  [ ! -s "${CURL_LOG}" ]
}

@test "the probe is bounded so an unresponsive host cannot hang the gate" {
  run smoke
  [ "$status" -eq 0 ]
  [[ "$(cat "${CURL_LOG}")" == *"--max-time"* ]]
}
