#!/usr/bin/env bats
# scripts/assert_installed_matches.sh

setup() {
  SCRIPT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)/scripts/assert_installed_matches.sh"
}

@test "identical files pass and the message names the hash" {
  printf 'same' > "${BATS_TEST_TMPDIR}/a"
  printf 'same' > "${BATS_TEST_TMPDIR}/b"
  run bash "${SCRIPT}" "${BATS_TEST_TMPDIR}/a" "${BATS_TEST_TMPDIR}/b"
  [ "$status" -eq 0 ]
  [[ "$output" == *"equals"* ]]
}

@test "identical files pass when the installed path contains a backslash" {
  mkdir "${BATS_TEST_TMPDIR}/a\\b"
  printf 'same' > "${BATS_TEST_TMPDIR}/a\\b/installed"
  printf 'same' > "${BATS_TEST_TMPDIR}/smoked"
  run bash "${SCRIPT}" "${BATS_TEST_TMPDIR}/a\\b/installed" "${BATS_TEST_TMPDIR}/smoked"
  [ "$status" -eq 0 ]
  [[ "$output" == *"equals"* ]]
}

@test "files that differ by one byte fail" {
  printf 'same' > "${BATS_TEST_TMPDIR}/a"
  printf 'sama' > "${BATS_TEST_TMPDIR}/b"
  run bash "${SCRIPT}" "${BATS_TEST_TMPDIR}/a" "${BATS_TEST_TMPDIR}/b"
  [ "$status" -ne 0 ]
  [[ "$output" == *"the smoked asset"* ]]
}

@test "a missing installed file is a usage error" {
  printf 'same' > "${BATS_TEST_TMPDIR}/b"
  run bash "${SCRIPT}" "${BATS_TEST_TMPDIR}/none" "${BATS_TEST_TMPDIR}/b"
  [ "$status" -ne 0 ]
  [[ "$output" == *"usage:"* ]]
}
