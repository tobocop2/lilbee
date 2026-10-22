#!/usr/bin/env bats
# scripts/check_flake_pins.sh refuses a flake sources file whose pins are malformed.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/check_flake_pins.sh"
  DIR="${BATS_TEST_TMPDIR}/pins"
  mkdir -p "${DIR}"
  GOOD=ff72ce42c64c8353172a7e4da7465c734d5ea4b2437e719bb9ec3240424468fb
}

pins() {  # pins <file> <json>
  printf '%s\n' "$2" > "${DIR}/$1"
}

check() {
  bash "${SCRIPT}" "${DIR}"/sources*.json
}

@test "well-formed pins pass" {
  pins sources.json "{\"version\":\"0.6.90b449\",\"systems\":{\"x86_64-linux\":{\"asset\":\"lilbee-linux-x86_64\",\"sha256\":\"${GOOD}\"}}}"
  run check
  [ "$status" -eq 0 ]
}

@test "a variant with no systems yet passes" {
  pins sources-cuda.json '{"version":"0.6.90b449","systems":{}}'
  run check
  [ "$status" -eq 0 ]
}

@test "a hash that is not 64 hex digits fails and names the file" {
  pins sources.json "{\"version\":\"1\",\"systems\":{\"x86_64-linux\":{\"asset\":\"a\",\"sha256\":\"not-a-hash\"}}}"
  pins sources-cuda.json '{"version":"1","systems":{}}'
  run check
  [ "$status" -eq 1 ]
  [[ "$output" == *"sources.json"*"x86_64-linux"* ]]
}

@test "a hash one digit short fails" {
  pins sources.json "{\"version\":\"1\",\"systems\":{\"x86_64-linux\":{\"asset\":\"a\",\"sha256\":\"${GOOD:1}\"}}}"
  run check
  [ "$status" -eq 1 ]
}

@test "a missing asset fails" {
  pins sources.json "{\"version\":\"1\",\"systems\":{\"x86_64-linux\":{\"sha256\":\"${GOOD}\"}}}"
  run check
  [ "$status" -eq 1 ]
}

@test "an empty version fails" {
  pins sources.json '{"version":"","systems":{}}'
  run check
  [ "$status" -eq 1 ]
}

@test "a file that is not JSON fails" {
  pins sources.json '{"version":'
  run check
  [ "$status" -eq 1 ]
}

@test "a file that is JSON but not an object is named as such" {
  for body in '[]' '"pins"' '3' 'null'; do
    pins sources.json "${body}"
    run check
    [ "$status" -eq 1 ]
    [[ "$output" == *"sources.json: not a JSON object"* ]] || { echo "${body}: ${output}" >&2; return 1; }
    [[ "$output" != *"not valid JSON"* ]]
  done
}

@test "no file to check fails" {
  run bash "${SCRIPT}"
  [ "$status" -eq 2 ]
}

@test "the repository's own pins pass" {
  run bash "${SCRIPT}" "${REPO_ROOT}"/sources*.json
  [ "$status" -eq 0 ]
  [ "$(printf '%s\n' "$output" | grep -c ': ok$')" -eq 4 ]
}

@test "the CI flake job runs this check on every sources file" {
  run yq -r '.jobs.flake.steps[] | select(.run != null) | .run' "${REPO_ROOT}/.github/workflows/ci.yml"
  [[ "$output" == *"bash scripts/check_flake_pins.sh sources*.json"* ]]
}
