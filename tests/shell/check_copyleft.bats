#!/usr/bin/env bats
# tools/ci/check_copyleft.py fails a package whose licence list holds a GPL or AGPL entry.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  cd "${REPO_ROOT}"
  uv export --locked --only-group license-check --no-hashes --no-annotate \
    --no-emit-project > "${BATS_TEST_TMPDIR}/checker.txt"
}

# $1 is the report JSON text.
copyleft() {
  printf '%s' "$1" > "${BATS_TEST_TMPDIR}/report.json"
  uv run --no-project --python 3.12 --with-requirements "${BATS_TEST_TMPDIR}/checker.txt" \
    python -I tools/ci/check_copyleft.py "${BATS_TEST_TMPDIR}/report.json" pyproject.toml
}

@test "a GPL entry next to a permissive entry fails and the package is named" {
  run copyleft '{"packages":[{"name":"plant-gpl","license":"GPLv3;; MIT","licenseCompat":true}]}'
  [ "$status" -eq 1 ]
  [[ "$output" == *"plant-gpl"* ]]
}

@test "an AGPL entry next to a permissive entry fails and the package is named" {
  run copyleft '{"packages":[{"name":"plant-agpl","license":"AGPL;; BSD","licenseCompat":true}]}'
  [ "$status" -eq 1 ]
  [[ "$output" == *"plant-agpl"* ]]
}

@test "permissive packages and an allow-listed package pass" {
  run copyleft '{"packages":[{"name":"plain","license":"MIT;; BSD","licenseCompat":true},{"name":"nvidia-cublas-cu12","license":"GPLv3;; MIT","licenseCompat":true}]}'
  [ "$status" -eq 0 ]
  [ -z "$output" ]
}
