#!/usr/bin/env bats
# tools/wheel-build/pack_common.sh: the functions that the pack scripts share.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  BIN="${BATS_TEST_TMPDIR}/bin"
  mkdir -p "${BIN}" "${BATS_TEST_TMPDIR}/assets"
  cp "${BATS_TEST_DIRNAME}/stubs/gh-pack" "${BIN}/gh"
  chmod +x "${BIN}/gh"
  PATH="${BIN}:${PATH}"
  export GH_ASSETS="${BATS_TEST_TMPDIR}/assets" GH_LOG="${BATS_TEST_TMPDIR}/gh.log"
  echo archive > "${GH_ASSETS}/a.tar.gz"
  echo other > "${GH_ASSETS}/b.tar.gz"
  # The scripts that source the file set these options.
  # shellcheck source=tools/wheel-build/pack_common.sh
  . "${REPO_ROOT}/tools/wheel-build/pack_common.sh"
  cd "${BATS_TEST_TMPDIR}"
}

# Run a function in a shell with the options that the pack scripts set.
strict() {
  bash -c 'set -euo pipefail; . "$0"; "$@"' "${REPO_ROOT}/tools/wheel-build/pack_common.sh" "$@"
}

# A PATH that holds the tools the functions need and no sha256sum.
without_sha256sum() {
  local tool
  mkdir -p "${BATS_TEST_TMPDIR}/bare"
  for tool in bash dirname basename shasum cat cp ls mkdir tr grep awk; do
    ln -s "$(command -v "${tool}")" "${BATS_TEST_TMPDIR}/bare/${tool}"
  done
  echo "${BATS_TEST_TMPDIR}/bare"
}

@test "fetch_release_asset downloads the named asset of the tag" {
  run strict fetch_release_asset owner/repo v1 a.tar.gz upstream
  [ "$status" -eq 0 ]
  [ "$(cat upstream/a.tar.gz)" = archive ]
  [ ! -e upstream/b.tar.gz ]
  grep -qx 'release download v1 -R owner/repo --pattern a.tar.gz --dir upstream' "${GH_LOG}"
  grep -q '^release view v1 -R owner/repo --json assets' "${GH_LOG}"
}

@test "fetch_release_asset fails when the digest differs" {
  GH_DIGEST=sha256:0000 run strict fetch_release_asset owner/repo v1 a.tar.gz upstream
  [ "$status" -eq 1 ]
  [[ "$output" == *"digest mismatch a.tar.gz: sha256:0000 sha256:"* ]]
}

@test "fetch_release_asset fails when GitHub lists no digest" {
  GH_DIGEST=null run strict fetch_release_asset owner/repo v1 a.tar.gz upstream
  [ "$status" -eq 1 ]
}

@test "fetch_release_asset fails when the release lacks the asset" {
  run strict fetch_release_asset owner/repo v1 c.tar.gz upstream
  [ "$status" -eq 1 ]
}

@test "fetch_release_asset fails when gh fails after it wrote the file" {
  GH_DOWNLOAD_STATUS=1 run strict fetch_release_asset owner/repo v1 a.tar.gz upstream
  [ "$status" -eq 1 ]
}

@test "fetch_release_asset fails in a condition too, where the shell ignores set -e" {
  GH_DIGEST=sha256:0000 run strict eval 'if fetch_release_asset owner/repo v1 a.tar.gz upstream; then echo passed; else echo refused; fi'
  [[ "$output" == *refused* ]]
  [[ "$output" != *passed* ]]
}

@test "sha256_line writes the digest and the file name without its folder" {
  run strict sha256_line assets/a.tar.gz
  [ "$status" -eq 0 ]
  [[ "$output" =~ ^[0-9a-f]{64}\ [\ *]a\.tar\.gz$ ]]
}

@test "sha256_line uses shasum where the computer has no sha256sum" {
  local line bare
  line="$(strict sha256_line assets/a.tar.gz)"
  bare="$(without_sha256sum)"
  PATH="${bare}" run strict sha256_line assets/a.tar.gz
  [ "$status" -eq 0 ]
  [ "${output%% *}" = "${line%% *}" ]
  [ "${output##*[ *]}" = a.tar.gz ]
}

@test "merge_terms keeps the mistral.rs order: MIT, the crates, then the other terms" {
  run strict eval '{ echo MIT; printf "0BSD\nApache-2.0\nBSD-3-Clause\nMIT\nZlib\n"; printf "Apache-2.0\nBSD-3-Clause\nLicenseRef-Microsoft-VC-Runtime\n"; } | merge_terms'
  [ "$status" -eq 0 ]
  [ "$output" = "MIT AND 0BSD AND Apache-2.0 AND BSD-3-Clause AND Zlib AND LicenseRef-Microsoft-VC-Runtime" ]
}

@test "merge_terms keeps the order of the cell before the sorted terms, as the llama.cpp and ik packs do" {
  run strict eval '{ printf "MIT\nLicenseRef-NVIDIA-CUDA-EULA\n"; printf "Unlicense\nMIT\nBSD-3-Clause\n" | sort -u; } | merge_terms'
  [ "$status" -eq 0 ]
  [ "$output" = "MIT AND LicenseRef-NVIDIA-CUDA-EULA AND BSD-3-Clause AND Unlicense" ]
}

@test "merge_terms drops carriage returns and empty lines" {
  run strict eval 'printf "MIT\r\n\r\n\nISC\r\nMIT\n" | merge_terms'
  [ "$status" -eq 0 ]
  [ "$output" = "MIT AND ISC" ]
}

@test "merge_terms keeps a term that holds a space as one term" {
  run strict eval 'printf "Apache-2.0 WITH LLVM-exception\nApache-2.0\n" | merge_terms'
  [ "$output" = "Apache-2.0 WITH LLVM-exception AND Apache-2.0" ]
}

@test "merge_terms fails without a term" {
  run strict eval 'printf "\n" | merge_terms'
  [ "$status" -ne 0 ]
}

@test "collect_wheel copies the wheel and writes its sha256 file" {
  mkdir dist
  echo wheel > dist/x-1-py3-none-any.whl
  run strict collect_wheel dist out
  [ "$status" -eq 0 ]
  [ "$(cat out/x-1-py3-none-any.whl)" = wheel ]
  (cd out && sha256sum -c x-1-py3-none-any.whl.sha256)
}

@test "collect_wheel fails on two wheels" {
  mkdir dist
  echo a > dist/a.whl
  echo b > dist/b.whl
  run strict collect_wheel dist out
  [ "$status" -eq 1 ]
  [[ "$output" == *"expected one wheel in out, found 2"* ]]
}

@test "collect_wheel fails on a wheel of an earlier build in the output folder" {
  mkdir dist out
  echo a > dist/a.whl
  echo b > out/b.whl
  run strict collect_wheel dist out
  [ "$status" -eq 1 ]
}

@test "collect_wheel fails without a wheel" {
  mkdir dist
  run strict collect_wheel dist out
  [ "$status" -ne 0 ]
}

@test "host_os reads the runner, then the system" {
  cp "${BATS_TEST_DIRNAME}/stubs/uname" "${BIN}/uname"
  [ "$(RUNNER_OS=Windows strict host_os)" = Windows ]
  unset RUNNER_OS
  local pair
  for pair in Linux:Linux Darwin:macOS MINGW64_NT-10.0:Windows MSYS_NT-10.0:Windows; do
    printf '#!/bin/sh\necho %s\n' "${pair%%:*}" > "${BIN}/uname"
    chmod +x "${BIN}/uname"
    [ "$(strict host_os)" = "${pair#*:}" ]
  done
  printf '#!/bin/sh\necho Plan9\n' > "${BIN}/uname"
  run strict host_os
  [ "$status" -eq 1 ]
}

@test "python_312 takes the first interpreter that is 3.12" {
  printf '#!/bin/sh\necho 3.11\n' > "${BIN}/python"
  printf '#!/bin/sh\necho 3.12\n' > "${BIN}/python3.12"
  chmod +x "${BIN}/python" "${BIN}/python3.12"
  [ "$(strict python_312)" = python3.12 ]
  printf '#!/bin/sh\necho 3.12\n' > "${BIN}/python"
  [ "$(strict python_312)" = python ]
}

@test "python_312 passes over an interpreter that fails" {
  printf '#!/bin/sh\nexit 127\n' > "${BIN}/python"
  printf '#!/bin/sh\necho 3.12\n' > "${BIN}/python3.12"
  chmod +x "${BIN}/python" "${BIN}/python3.12"
  [ "$(strict python_312)" = python3.12 ]
}

@test "python_312 fails without a 3.12" {
  local name
  for name in python python3.12 python3; do
    printf '#!/bin/sh\necho 3.13\n' > "${BIN}/${name}"
    chmod +x "${BIN}/${name}"
  done
  run strict python_312
  [ "$status" -eq 1 ]
  [[ "$output" == *"Python 3.12 is not on PATH"* ]]
}
