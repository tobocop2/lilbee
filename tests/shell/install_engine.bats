#!/usr/bin/env bats
# scripts/install_engine.sh against a stubbed release download.

bats_require_minimum_version 1.5.0

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/install_engine.sh"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  mkdir -p "${FIXTURE}/bin"
  cp "${BATS_TEST_DIRNAME}/stubs/gh-engine" "${FIXTURE}/bin/gh"
  for stub in unzip tar sleep; do
    cp "${BATS_TEST_DIRNAME}/stubs/${stub}" "${FIXTURE}/bin/${stub}"
  done
  chmod +x "${FIXTURE}"/bin/*
  DEST="${FIXTURE}/engine"
  # The stub gh builds this tree in place of unpacking a real archive.
  ENGINE_PAYLOAD="${FIXTURE}/payload"
  GH_LOG="${FIXTURE}/gh-args"
  SLEEP_LOG="${FIXTURE}/sleeps"
  : > "${SLEEP_LOG}"
}

# The script traces itself, so stdout carries the directory alone and every
# message the assertions read is on stderr.
install_engine() {
  GH_FAIL_TIMES="${GH_FAIL_TIMES:-0}" \
  GH_NO_ASSET="${GH_NO_ASSET:-}" \
  ENGINE_PAYLOAD="${ENGINE_PAYLOAD}" \
  GH_LOG="${GH_LOG}" \
  SLEEP_LOG="${SLEEP_LOG}" \
  PATH="${FIXTURE}/bin:${PATH}" \
  bash "${SCRIPT}" "$@"
}

# A payload directory the stub gh packs into the archive it is asked for.
payload_with_engine() {
  mkdir -p "${ENGINE_PAYLOAD}/build/bin"
  printf '#!/bin/sh\n' > "${ENGINE_PAYLOAD}/build/bin/${1}"
  chmod +x "${ENGINE_PAYLOAD}/build/bin/${1}"
}

payload_without_engine() {
  mkdir -p "${ENGINE_PAYLOAD}/build/bin"
  printf 'no engine here\n' > "${ENGINE_PAYLOAD}/build/bin/README.md"
}

@test "a linux archive yields the directory holding llama-server" {
  payload_with_engine llama-server
  run --separate-stderr install_engine b9351 Linux "${DEST}"
  [ "$status" -eq 0 ]
  [ "$output" = "${DEST}/build/bin" ]
  [ -x "${output}/llama-server" ]
  # The archive is unpacked from outside the destination, not into it.
  [ "$(ls "${DEST}")" = build ]
}

@test "a windows archive yields the directory holding llama-server.exe" {
  payload_with_engine llama-server.exe
  run --separate-stderr install_engine b9351 Windows "${DEST}"
  [ "$status" -eq 0 ]
  [ "$output" = "${DEST}/build/bin" ]
}

# The suffix is anchored, so the plain macOS asset is asked for and the
# -kleidiai variant that shares its prefix is not.
@test "the macos asset glob ends at the archive suffix" {
  payload_with_engine llama-server
  run --separate-stderr install_engine b9351 macOS "${DEST}"
  [ "$status" -eq 0 ]
  grep -qF -- "--pattern *bin-macos-arm64.tar.gz " "${GH_LOG}"
}

# A download that fails must not leave the caller with a directory name that
# looks usable.
@test "a download that always fails stops instead of naming a directory" {
  payload_with_engine llama-server
  GH_FAIL_TIMES=99 run --separate-stderr install_engine b9351 Windows "${DEST}"
  [ "$status" -ne 0 ]
  [ -z "$output" ]
}

# `dirname ''` is '.', so an empty find must not become a directory name.
@test "an archive without an engine stops instead of returning a dot" {
  payload_without_engine
  run --separate-stderr install_engine b9351 Linux "${DEST}"
  [ "$status" -eq 1 ]
  [ -z "$output" ]
  [[ "$stderr" == *"no llama-server in"* ]]
}

@test "a transient download failure is retried and then succeeds" {
  payload_with_engine llama-server
  GH_FAIL_TIMES=1 run --separate-stderr install_engine b9351 Linux "${DEST}"
  [ "$status" -eq 0 ]
  [ "$output" = "${DEST}/build/bin" ]
  [ "$(wc -l < "${SLEEP_LOG}")" -eq 1 ]
}

@test "a release with no matching asset stops before extracting" {
  payload_with_engine llama-server
  GH_NO_ASSET=1 run --separate-stderr install_engine b9351 Linux "${DEST}"
  [ "$status" -ne 0 ]
  [ -z "$output" ]
  [[ "$stderr" == *"no assets match the file pattern"* ]]
}

@test "an unknown runner os is rejected" {
  run --separate-stderr install_engine b9351 Plan9 "${DEST}"
  [ "$status" -eq 2 ]
  [[ "$stderr" == *"no asset pattern"* ]]
}

@test "a missing argument is rejected" {
  run --separate-stderr install_engine b9351 Linux
  [ "$status" -eq 2 ]
  [[ "$stderr" == *"usage"* ]]
}
