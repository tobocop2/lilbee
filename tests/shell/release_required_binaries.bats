#!/usr/bin/env bats
# scripts/assert_required_binaries.sh, the gate that decides whether a dropped
# build cell may strand the release.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/assert_required_binaries.sh"
  WORKFLOW="${REPO_ROOT}/.github/workflows/release.yml"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  mkdir -p "${FIXTURE}/bins" "${FIXTURE}/held"
  # The cells release.yml carries without `soft: true`.
  REQUIRED=(lilbee-linux-x86_64 lilbee-macos-arm64 lilbee-windows-x86_64.exe)
  for asset in "${REQUIRED[@]}"; do : > "${FIXTURE}/bins/${asset}"; done
  : > "${FIXTURE}/bins/lilbee-compat-windows-x86_64.exe"
}

assert_binaries() {
  bash "${SCRIPT}" "${FIXTURE}/bins" "${WORKFLOW}"
}

@test "a complete set of executables passes" {
  run assert_binaries
  [ "$status" -eq 0 ]
  [[ "$output" == *"every required executable is present"* ]]
}

@test "a soft cell that dropped out does not block the release" {
  rm "${FIXTURE}/bins/lilbee-compat-windows-x86_64.exe"
  run assert_binaries
  [ "$status" -eq 0 ]
}

@test "every cell without soft is required, and the gate names the one missing" {
  [ "${#REQUIRED[@]}" -eq 3 ]
  local checked=0
  for asset in "${REQUIRED[@]}"; do
    mv "${FIXTURE}/bins/${asset}" "${FIXTURE}/held/${asset}"
    run assert_binaries
    [ "$status" -eq 1 ]
    [[ "$output" == *"never reached"* ]]
    [[ "$output" == *"${asset}"* ]]
    mv "${FIXTURE}/held/${asset}" "${FIXTURE}/bins/${asset}"
    checked=$(( checked + 1 ))
  done
  [ "${checked}" -eq 3 ]
}

@test "a matrix the derivation cannot read fails loudly" {
  printf 'jobs:\n  build:\n    runs-on: ubuntu-latest\n' > "${FIXTURE}/empty.yml"
  run bash "${SCRIPT}" "${FIXTURE}/bins" "${FIXTURE}/empty.yml"
  [ "$status" -eq 1 ]
  [[ "$output" == *"no required cell"* ]]
}
