#!/usr/bin/env bats
# The engine cache run must fail when a backend cell fails.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  BUILD="${REPO_ROOT}/.github/workflows/build-multigpu.yml"
  WARM="${REPO_ROOT}/.github/workflows/warm-engine-cache.yml"
}

@test "the engine cache run builds every backend as required" {
  run yq -r '.jobs.warm.with.required' "${WARM}"
  [ "$status" -eq 0 ]
  [ "$output" = "true" ]
}

@test "a backend cell is droppable only when the caller did not ask for required" {
  run yq -r '.jobs["build-multigpu"]["continue-on-error"]' "${BUILD}"
  [ "$status" -eq 0 ]
  [ "$output" = '${{ !inputs.required }}' ]
}

@test "the mirror upload is the only step that swallows a failure" {
  run yq -r '[.jobs["build-multigpu"].steps[] | select(.["continue-on-error"] == true) | .name] | join("|")' "${BUILD}"
  [ "$status" -eq 0 ]
  [ "$output" = "Mirror the built engine" ]
}

@test "the engine cache run starts for an engine pin bump before it merges" {
  # yq reads the bare key `on` as a boolean, so the triggers are read as text.
  run grep -c -E '^  (pull_request|push):$' "${WARM}"
  [ "$output" = "2" ]
  run grep -F 'branches: [main, chore/engine-pin-sync]' "${WARM}"
  [ "$status" -eq 0 ]
}
