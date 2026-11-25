#!/usr/bin/env bats
# The engine cache run must fail when a backend cell fails.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  BUILD="${REPO_ROOT}/.github/workflows/build-multigpu.yml"
  WARM="${REPO_ROOT}/.github/workflows/warm-engine-cache.yml"
  CI="${REPO_ROOT}/.github/workflows/ci.yml"
}

# The paths of the CI run's engine filter, one per line.
engine_paths() {
  yq -r '.jobs.changes.steps[] | select(.id == "filter") | .with.filters' "${CI}" \
    | yq -r '.engine[]'
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

# Branch protection requires ci-result alone, so an engine build outside the CI
# run reports a red check that does not block the merge.
@test "the required CI run compiles every engine backend as required" {
  run yq -r '.jobs.engine.uses' "${CI}"
  [ "$status" -eq 0 ]
  [ "$output" = "./.github/workflows/build-multigpu.yml" ]
  run yq -r '.jobs.engine.with.required' "${CI}"
  [ "$output" = "true" ]
}

@test "the CI engine build runs when an engine path changed" {
  run yq -r '.jobs.engine.if' "${CI}"
  [ "$status" -eq 0 ]
  [ "$output" = "needs.changes.outputs.engine == 'true'" ]
  run yq -r '.jobs.changes.outputs.engine' "${CI}"
  [ "$output" = "\${{ steps.filter.outputs.engine || 'true' }}" ]
}

@test "ci-result judges the engine build" {
  run yq -r '.jobs["ci-result"].needs | index("engine")' "${CI}"
  [ "$status" -eq 0 ]
  [ "$output" != "null" ]
}

@test "the engine paths are the engine key's files and the build workflow" {
  run engine_paths
  [ "$status" -eq 0 ]
  paths="$output"
  run hashed_files "${BUILD}"
  expected=$(printf '%s\n' "${output}" '.github/workflows/build-multigpu.yml' | sort)
  [ "$(echo "${paths}" | sort)" = "${expected}" ]
}

@test "the weekly engine cache run does not build a second time on a change" {
  # yq reads the bare key `on` as a boolean, so the triggers are read as text.
  run grep -c -E '^  (pull_request|push):$' "${WARM}"
  [ "$output" = "0" ]
}

# The hashFiles list of the llama.cpp engine key, one path per line. The Vulkan SDK key hashes
# engine-versions.env alone and is not an engine key.
hashed_files() {
  grep -o "hashFiles('engine-versions.env', [^)]*)" "$1" | grep -o "'[^']*'" | tr -d "'"
}

@test "both llama.cpp engine keys hash the same files and not the OpenVINO pin" {
  ACTION="${REPO_ROOT}/.github/actions/bundle-llama-server/action.yml"
  run hashed_files "${BUILD}"
  [ "$status" -eq 0 ]
  build_list="$output"
  run hashed_files "${ACTION}"
  [ "$status" -eq 0 ]
  [ "$output" = "${build_list}" ]
  echo "${build_list}" | grep -qxF 'engine-versions.env'
  ! echo "${build_list}" | grep -qxF 'engines/openvino.env'
}

@test "an OpenVINO pin bump does not start the engine build" {
  run engine_paths
  [ "$status" -eq 0 ]
  ! echo "${output}" | grep -qxF 'engines/openvino.env'
}
