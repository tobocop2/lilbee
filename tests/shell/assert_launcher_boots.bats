#!/usr/bin/env bats
# scripts/assert_launcher_boots.sh against a stubbed lilbee executable.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/assert_launcher_boots.sh"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  mkdir -p "${FIXTURE}"
  EXE="${FIXTURE}/lilbee"
}

# A stub executable. --version prints the given stdout and stderr and exits
# with the given code; --help prints the given stderr and exits with its own
# code, defaulting to clean.
stub_exe() {  # version-stdout version-stderr version-rc [help-stderr] [help-rc]
  cat > "${EXE}" <<STUB
#!/usr/bin/env bash
case "\$1" in
  --version)
    printf '%s\n' "$1"
    printf '%s\n' "$2" >&2
    exit $3
    ;;
  --help)
    printf 'Usage: lilbee\n'
    printf '%s\n' "${4:-}" >&2
    exit ${5:-0}
    ;;
esac
STUB
  chmod +x "${EXE}"
}

# The gate as it stood before the fix: the probe runs in a command
# substitution under set -e, and its stderr is printed on the following line.
old_gate() {  # expected-version
  bash -c '
    set -euo pipefail
    STDERR=$(mktemp)
    ACTUAL=$("$1" --version 2>"$STDERR")
    echo "Actual:   ${ACTUAL}"
    echo "Stderr:"
    cat "$STDERR"
    test "${ACTUAL}" = "lilbee $2"
  ' _ "${EXE}" "$1"
}

@test "control: the old gate discards the diagnostic when the probe fails" {
  stub_exe "" "dyld: Library not loaded libllama.dylib" 1
  run old_gate 1.2.3
  [ "$status" -ne 0 ]
  [[ "$output" != *"Library not loaded"* ]]
}

@test "a failing probe prints its exit code and stderr, then fails" {
  stub_exe "" "dyld: Library not loaded libllama.dylib" 1
  run bash "${SCRIPT}" "${EXE}" 1.2.3
  [ "$status" -eq 1 ]
  [[ "$output" == *"--version exit code: 1"* ]]
  [[ "$output" == *"Library not loaded"* ]]
  [[ "$output" == *"FAIL: --version exited 1"* ]]
}

@test "a version mismatch prints both versions, then fails" {
  stub_exe "lilbee 9.9.9" "" 0
  run bash "${SCRIPT}" "${EXE}" 1.2.3
  [ "$status" -eq 1 ]
  [[ "$output" == *"Expected: lilbee 1.2.3"* ]]
  [[ "$output" == *"Actual:   lilbee 9.9.9"* ]]
  [[ "$output" == *"FAIL: --version printed the wrong version"* ]]
}

@test "the right version printed on a non-zero exit still fails" {
  stub_exe "lilbee 1.2.3" "dyld: Library not loaded libllama.dylib" 1
  run bash "${SCRIPT}" "${EXE}" 1.2.3
  [ "$status" -eq 1 ]
  [[ "$output" == *"Library not loaded"* ]]
  [[ "$output" == *"FAIL: --version exited 1"* ]]
}

@test "a failing --help prints its exit code and stderr, then fails" {
  stub_exe "lilbee 1.2.3" "" 0 "Segmentation fault" 139
  run bash "${SCRIPT}" "${EXE}" 1.2.3
  [ "$status" -eq 1 ]
  [[ "$output" == *"--help exit code: 139"* ]]
  [[ "$output" == *"Segmentation fault"* ]]
  [[ "$output" == *"FAIL: --help exited 139"* ]]
}

@test "a typer leak on --help stderr fails even when --help exits clean" {
  stub_exe "lilbee 1.2.3" "" 0 "No such option: -B" 0
  run bash "${SCRIPT}" "${EXE}" 1.2.3
  [ "$status" -eq 1 ]
  [[ "$output" == *"No such option: -B"* ]]
  [[ "$output" == *"FAIL: --help produced typer/runtime errors on stderr"* ]]
}

@test "a typer leak on stderr fails even when the version matches" {
  stub_exe "lilbee 1.2.3" "No such option: -B" 0
  run bash "${SCRIPT}" "${EXE}" 1.2.3
  [ "$status" -eq 1 ]
  [[ "$output" == *"FAIL: --version produced typer/runtime errors on stderr"* ]]
}

@test "a clean launcher passes" {
  stub_exe "lilbee 1.2.3" "" 0
  run bash "${SCRIPT}" "${EXE}" 1.2.3
  [ "$status" -eq 0 ]
  [[ "$output" == *"LAUNCHER OK"* ]]
}
