#!/usr/bin/env bats
# scripts/release_plan_cells.sh: the cells the release build job runs, narrowed
# to one asset when a pull request builds a single executable.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/release_plan_cells.sh"
  CELLS="${REPO_ROOT}/.github/release-cells.yml"
}

@test "no asset selects every cell" {
  run bash "${SCRIPT}" "" "${CELLS}"
  [ "$status" -eq 0 ]
  [ "$(jq 'length' <<< "$output")" -eq "$(yq -r 'length' "${CELLS}")" ]
}

@test "an asset selects only its cell" {
  run bash "${SCRIPT}" lilbee-windows-x86_64.exe "${CELLS}"
  [ "$status" -eq 0 ]
  [ "$(jq -r 'length' <<< "$output")" -eq 1 ]
  [ "$(jq -r '.[0].os' <<< "$output")" = "windows-latest" ]
}

@test "a cell keeps its optional fields as JSON values" {
  run bash "${SCRIPT}" lilbee-compat-macos-x86_64 "${CELLS}"
  [ "$status" -eq 0 ]
  [ "$(jq -r '.[0].soft' <<< "$output")" = "true" ]
  [ "$(jq -r '.[0].macos_floor' <<< "$output")" = "12.0" ]
}

@test "the output is one line, which a step output requires" {
  run bash "${SCRIPT}" "" "${CELLS}"
  [ "$status" -eq 0 ]
  [ "${#lines[@]}" -eq 1 ]
}

@test "an asset no cell builds fails instead of selecting nothing" {
  run bash "${SCRIPT}" lilbee-no-such-asset "${CELLS}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"no build cell"* ]]
}
