#!/usr/bin/env bats
# tools/wheel-build/pack_tei.sh in a copy of the tree, with stubs for docker, the Rust tools, delocate and the Python
# build tools. The wheel check and the smoke test are the real ones, on a wheel whose router is a shell script.
# tests/test_tei_wheel_tools.py tests tei_manylinux.sh, tei_wheel.sh, check_tei_wheel.py and smoke_tei.sh.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  STUBS="${BATS_TEST_DIRNAME}/stubs"
  ROOT="${BATS_TEST_TMPDIR}/lil bee"
  BIN="${BATS_TEST_TMPDIR}/bin"
  mkdir -p "${ROOT}/packaging" "${ROOT}/tools" "${BIN}" "${BATS_TEST_TMPDIR}/tmp"
  cp -R "${REPO_ROOT}/tools/wheel-build" "${ROOT}/tools/"
  cp -R "${REPO_ROOT}/packaging/engine-wheel-tei" "${ROOT}/packaging/"
  SCRIPT="${ROOT}/tools/wheel-build/pack_tei.sh"
  TREE="${ROOT}/packaging/engine-wheel-tei"

  REAL_PYTHON="$(command -v python3)"
  cp "${STUBS}/python-tei-pack" "${BIN}/python"
  cp "${STUBS}/docker-tei-pack" "${BIN}/docker"
  cp "${STUBS}/tei-wheel-pack" "${STUBS}/uname" "${BIN}/"
  # The real qemu and ldd do not read a shell script.
  tool qemu-x86_64-static 'echo "$*" >> "${QEMU_LOG}"' '[ -z "${QEMU_FAIL-}" ] || exit 132' 'shift 2' 'exec "$@"'
  tool ldd 'echo "${LDD_TEXT:-libc.so.6 => /lib/libc.so.6}"'
  tool curl 'case "$*" in */embed*) echo "[[$(yes 0 | head -n 384 | paste -s -d, -)]]" ;; esac'
  PATH="${BIN}:${PATH}"

  export REAL_PYTHON DOCKER_LOG="${BATS_TEST_TMPDIR}/docker.log" PYTHON_LOG="${BATS_TEST_TMPDIR}/python.log" PYTHON_CALLS="${BATS_TEST_TMPDIR}/calls.log"
  export QEMU_LOG="${BATS_TEST_TMPDIR}/qemu.log" ROUTER_LOG="${BATS_TEST_TMPDIR}/router.log" ROUTER_FAIL=
  export STEP_LOG="${BATS_TEST_TMPDIR}/step.log" TMPDIR="${BATS_TEST_TMPDIR}/tmp/"
  export TEI_REPO="${BATS_TEST_TMPDIR}/tei-origin" TEI_TAG=v1.9.4 RUSTC_VERSION=1.92.0 WHEEL_VERSION=1.9.4
  export CUDA_VERSION=12.9.1 CUDA_PACKAGE_SUFFIX=12-9 CUDA_CC= BACKEND=cpu PLATFORM=manylinux_2_34_x86_64
  export MANYLINUX_IMAGE=quay.io/pypa/x@sha256:abc RUNNER_OS=Linux OUT="${BATS_TEST_TMPDIR}/packs-out"
  unset SOURCE_DATE_EPOCH EMULATE_CPU STUB_DIR PIP_CONSTRAINT CARGO_HOME RUSTFLAGS
  WHEEL=lilbee_engine_tei-1.9.4-1.cpu-py3-none-manylinux_2_34_x86_64.whl
  cd "${BATS_TEST_TMPDIR}"
}

# Write one tool of BIN from its lines.
tool() {  # name line...
  local name="$1"
  shift
  printf '%s\n' '#!/usr/bin/env bash' "$@" > "${BIN}/${name}"
  chmod +x "${BIN}/${name}"
}

cuda() {
  export BACKEND=cuda-sm80 CUDA_CC=80
  WHEEL=lilbee_engine_tei-1.9.4-1.cuda_sm80-py3-none-manylinux_2_34_x86_64.whl
}

# Set up the macOS cell. The Rust and Mac tools append to STEP_LOG, in the order of the calls.
metal() {
  export BACKEND=metal PLATFORM=macosx_14_0_arm64 MACOSX_DEPLOYMENT_TARGET=14.0 RUNNER_OS=macOS UNAME_S=Darwin
  WHEEL=lilbee_engine_tei-1.9.4-1.metal-py3-none-macosx_14_0_arm64.whl
  WORK="${BATS_TEST_TMPDIR}/tmp/lilbee-tei-v1.9.4"
  SRC="${WORK}/src"
  tool otool 'echo /usr/lib/libSystem.B.dylib'
  tool xcodebuild 'echo "Xcode 26.0"'
  tool rustup 'if [ "$*" = "run 1.92.0 rustc -vV" ]; then printf "rustc 1.92.0\ncommit-hash: %040d\nrelease: 1.92.0\n" 192; exit; fi' \
    'echo "rustup $*" >> "${STEP_LOG}"'
  # The router holds the commit of each compiler in COMPILERS, as the routers of rustc do.
  tool cargo 'echo "cargo $* | flags=${RUSTFLAGS-} toolchain=${RUSTUP_TOOLCHAIN-} home=${CARGO_HOME-} target=${CARGO_TARGET_DIR-unset} encoded=${CARGO_ENCODED_RUSTFLAGS-unset} wrapper=${RUSTC_WRAPPER-unset} rustc=${RUSTC-unset} workspace=${RUSTC_WORKSPACE_WRAPPER-unset} floor=${MACOSX_DEPLOYMENT_TARGET-} cwd=${PWD}" >> "${STEP_LOG}"' \
    '[ -z "${COMPILE_FAIL-}" ] || exit 101' 'mkdir -p target/release' 'exec > target/release/text-embeddings-router' 'printf "router\0"' \
    'for commit in ${COMPILERS-192 192}; do printf "/rustc/%040d/library/core\0" "${commit}"; done'
  tool delocate-wheel 'echo "delocate $* | epoch=${SOURCE_DATE_EPOCH-unset}" >> "${STEP_LOG}"' '[ -z "${DELOCATE_FAIL-}" ] || exit 1' \
    'mkdir -p out/repaired' 'cp out/dist/*.whl out/repaired/' \
    '[ -z "${GRAFT-}" ] || (cd out/repaired && "${REAL_PYTHON}" -c "import glob, zipfile; zipfile.ZipFile(glob.glob(\"*.whl\")[0], \"a\").writestr(\"lilbee_engine_tei/.dylibs/libx.dylib\", \"x\")")'
  printf '%s\n' 'echo "cargo-about | home=${CARGO_HOME-}" >> "${STEP_LOG}"' 'echo about > "$1/cargo-about"' > "${ROOT}/tools/wheel-build/install_cargo_about.sh"
  printf '%s\n' 'test -s "$2"' 'test "$(cat "${1%/src}/bin/cargo-about")" = about' \
    '(cd packaging/engine-wheel-tei && for file in ${PIP_CONSTRAINT}; do test -s "${file}"; done)' \
    'echo "wheel ${*:3} | src=$1 router=$2 python=${PYTHON} locale=${LC_ALL-} path=${PATH%%:*}" >> "${STEP_LOG}"' \
    'tei-wheel-pack "$7/dist" "$7/license-expression.txt" "$6"' > "${ROOT}/tools/wheel-build/tei_wheel.sh"
  mkdir -p "${TEI_REPO}"
  printf '[toolchain]\nchannel = "1.92.0"\n' > "${TEI_REPO}/rust-toolchain.toml"
  git -C "${TEI_REPO}" init -q -b main
  git -C "${TEI_REPO}" add rust-toolchain.toml
  # The tag is not the head of the default branch, and a later tag exists.
  source_commit 1700000000
  git -C "${TEI_REPO}" tag v1.9.4
  source_commit 1700000001
  git -C "${TEI_REPO}" tag v1.9.5
  source_commit 1700000002
}

source_commit() {  # commit time
  GIT_COMMITTER_DATE="$1 +0000" git -C "${TEI_REPO}" -c user.name=t -c user.email=t@example.test -c commit.gpgsign=false commit -q --allow-empty -m source
}

@test "the wheel, its sha256 file and the licence expression reach the output folder" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(ls "${OUT}")" = "$(printf '%s\n' license-expression.txt "${WHEEL}" "${WHEEL}.sha256" | sort)" ]
  (cd "${OUT}" && sha256sum -c "${WHEEL}.sha256")
  [ "$(cat "${OUT}/license-expression.txt")" = "Apache-2.0 AND MIT" ]
}

@test "every variable of the contract is required" {
  local name
  for name in TEI_REPO TEI_TAG RUSTC_VERSION WHEEL_VERSION BACKEND PLATFORM OUT CUDA_VERSION CUDA_PACKAGE_SUFFIX MANYLINUX_IMAGE; do
    run env "${name}=" bash "${SCRIPT}"
    [ "$status" -eq 2 ]
    [[ "$output" == *"${name} is not set"* ]]
  done
  [ ! -e "${DOCKER_LOG}" ]
  metal
  run env MACOSX_DEPLOYMENT_TARGET= bash "${SCRIPT}"
  [ "$status" -eq 2 ]
  [[ "$output" == *"MACOSX_DEPLOYMENT_TARGET is not set"* ]]
}

@test "a computer without the emulator is refused before the build" {
  local bare="${BATS_TEST_TMPDIR}/bare" name
  mkdir -p "${bare}"
  for name in bash dirname uname git docker; do ln -s "$(command -v "${name}")" "${bare}/${name}"; done
  PATH="${bare}" run "${bare}/bash" "${SCRIPT}"
  [ "$status" -eq 2 ]
  [[ "$output" == *"qemu-x86_64-static is not installed"* ]]
  [ ! -e "${DOCKER_LOG}" ]
}

@test "a Mac without one of the build tools is refused before the build" {
  metal
  local bare name tool
  for name in git rustup jq xcodebuild; do
    bare="${BATS_TEST_TMPDIR}/bare-${name}"
    mkdir -p "${bare}"
    for tool in bash dirname; do ln -s "$(command -v "${tool}")" "${bare}/${tool}"; done
    for tool in git rustup jq xcodebuild; do
      [ "${tool}" = "${name}" ] || ln -s "${BIN}/rustup" "${bare}/${tool}"
    done
    PATH="${bare}" run "${bare}/bash" "${SCRIPT}"
    [ "$status" -eq 2 ]
    [[ "$output" == *"pack_tei: ${name} is not installed"* ]]
  done
  [ ! -e "${STEP_LOG}" ]
}

@test "Windows builds no cell" {
  RUNNER_OS=Windows run bash "${SCRIPT}"
  [ "$status" -eq 2 ]
  [[ "$output" == *"no cell builds on Windows"* ]]
}

@test "a checkout that an earlier build used is refused" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  run bash "${SCRIPT}"
  [ "$status" -eq 2 ]
  [[ "$output" == *"an earlier build used this checkout"* ]]
}

@test "each folder that an earlier build leaves is refused" {
  local name
  for name in out wheel pack-venv check-venv; do
    mkdir -p "${ROOT}-${name}"
    cp -R "${ROOT}/." "${ROOT}-${name}/"
    mkdir "${ROOT}-${name}/${name}"
    run bash "${ROOT}-${name}/tools/wheel-build/pack_tei.sh"
    [ "$status" -eq 2 ]
    [[ "$output" == *"(${name} exists)"* ]]
  done
  [ ! -e "${DOCKER_LOG}" ]
}

@test "a Python other than 3.12 is refused before the build" {
  local name
  for name in python3.12 python3; do cp "${BIN}/python" "${BIN}/${name}"; done
  PYTHON_VERSION=3.13 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"Python 3.12 is not on PATH"* ]]
  [ ! -e "${DOCKER_LOG}" ]
}

@test "a Linux cell builds in the pinned image with the checkout at /work" {
  cuda
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  local want
  want="$(printf '%s\n' run --rm -v "${ROOT}:/work" -w /work)"
  local name
  for name in TEI_REPO TEI_TAG RUSTC_VERSION WHEEL_VERSION BACKEND CUDA_CC CUDA_PACKAGE_SUFFIX CUDA_VERSION; do
    want="${want}
-e
${name}"
  done
  want="${want}
-e
PIP_CONSTRAINT=/work/tools/wheel-build/build-tools.constraints /work/tools/wheel-build/tei-tools.constraints
-e
WORK=/work
-e
OUT=/work/out
-e
OWNER=$(id -u):$(id -g)
${MANYLINUX_IMAGE}
bash
/work/tools/wheel-build/tei_manylinux.sh"
  [ "$(cat "${DOCKER_LOG}")" = "${want}" ]
}

@test "a build that fails stops the pack" {
  DOCKER_FAIL=1 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [ ! -e "${OUT}" ]
}

@test "two wheels stop the pack" {
  WHEEL_COUNT=2 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"expected one wheel"* ]]
  [ ! -e "${OUT}" ]
}

@test "a wheel with a wrong tag in its name is refused" {
  local name
  for name in \
    lilbee_engine_tei-1.9.4-1.cpu-py3-none-linux_x86_64.whl \
    lilbee_engine_tei-1.9.4-1.cpu-py3-none-manylinux_2_34_x86_64.manylinux_2_28_x86_64.whl \
    lilbee_engine_tei-1.9.4-1.metal-py3-none-manylinux_2_34_x86_64.whl \
    lilbee_engine_tei-1.9.5-1.cpu-py3-none-manylinux_2_34_x86_64.whl \
    lilbee_engine_tei-1.9.4-1.cpu-cp312-none-manylinux_2_34_x86_64.whl; do
    mkdir -p "${ROOT}-${name}"
    cp -R "${ROOT}/." "${ROOT}-${name}/"
    WHEEL_NAME="${name}" run bash "${ROOT}-${name}/tools/wheel-build/pack_tei.sh"
    [ "$status" -eq 1 ]
    [[ "$output" == *"wrong tag on ${name}"* ]]
  done
  [ ! -e "${OUT}" ]
  [ ! -e "${ROUTER_LOG}" ]
}

@test "a wheel with a wrong tag in its WHEEL file is refused" {
  WHEEL_TAG=linux_x86_64 run bash "${SCRIPT}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"lacks the tag py3-none-manylinux_2_34_x86_64"* ]]
  [ ! -e "${OUT}" ]
}

@test "a wheel with CRLF line ends in METADATA and WHEEL passes" {
  WHEEL_CRLF=1 run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ -f "${OUT}/${WHEEL}" ]
  python -m zipfile -e "${OUT}/${WHEEL}" "${BATS_TEST_TMPDIR}/wheel"
  grep -q $'^Tag: .*\r$' "${BATS_TEST_TMPDIR}/wheel/lilbee_engine_tei-1.9.4.dist-info/WHEEL"
  grep -q $'^License-Expression: .*\r$' "${BATS_TEST_TMPDIR}/wheel/lilbee_engine_tei-1.9.4.dist-info/METADATA"
}

@test "a wrong tag in a WHEEL file with CRLF line ends is refused" {
  WHEEL_CRLF=1 WHEEL_TAG=linux_x86_64 run bash "${SCRIPT}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"lacks the tag py3-none-manylinux_2_34_x86_64"* ]]
}

@test "the build tools go into a venv of the checkout" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ -x "${ROOT}/pack-venv/bin/python" ]
  [ "$(grep -c '^-m venv pack-venv$' "${PYTHON_LOG}")" -eq 1 ]
}

# Pass when each Python call after the venv of the checkout runs an interpreter of a venv, and each call in "$@" is one.
only_venv_python() {  # call...
  local call
  sed '1,/ -m venv pack-venv$/d' "${PYTHON_CALLS}" > calls.after
  for call in "$@"; do grep -qF -- "${ROOT}/pack-venv/bin/python ${call}" calls.after; done
  run grep -vF -e "${ROOT}/pack-venv/bin/python " -e 'check-venv/bin/python ' calls.after
  [ "$status" -eq 1 ]
}

@test "a Linux cell checks its wheel with the Python of the build venv" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  only_venv_python '-c import pathlib' 'tools/wheel-build/check_tei_wheel.py ' '-m venv check-venv'
}

@test "the macOS cell builds and checks its wheel with the Python of the build venv" {
  metal
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  only_venv_python '-m pip install -q build wheel delocate' '-m zipfile -l ' 'tools/wheel-build/check_tei_wheel.py ' '-m venv check-venv'
}

@test "a wheel that lacks a licence is refused" {
  cuda
  local name
  for name in LICENSE-tei LICENSE-openssl THIRD-PARTY/tei/LICENSES.txt NOTICE-NVIDIA-CUDA cutlass/LICENSE.txt; do
    mkdir -p "${ROOT}-${name##*/}"
    cp -R "${ROOT}/." "${ROOT}-${name##*/}/"
    WHEEL_WITHOUT="${name}" run bash "${ROOT}-${name##*/}/tools/wheel-build/pack_tei.sh"
    [ "$status" -eq 1 ]
    [[ "$output" == *"licen"* ]]
  done
  [ ! -e "${OUT}" ]
  [ ! -e "${ROUTER_LOG}" ]
}

@test "a router that does not start is refused" {
  cuda
  ROUTER_FAIL=1 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [ "$(cat "${ROUTER_LOG}")" = "router --version" ]
  [ ! -e "${OUT}" ]
}

@test "a router with an unresolved library is refused" {
  cuda
  LDD_TEXT='libssl.so.3 => not found' run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"unresolved library"* ]]
  [ ! -e "${OUT}" ]
}

@test "a Linux router that dies on the emulated CPU is refused" {
  cuda
  QEMU_FAIL=1 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [ ! -e "${OUT}" ]
}

@test "a Linux router starts on this CPU and on an emulated CPU without AVX" {
  cuda
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(cat "${ROUTER_LOG}")" = "router --version
router --version" ]
  [ "$(wc -l < "${QEMU_LOG}")" -eq 1 ]
  [[ "$(cat "${QEMU_LOG}")" == "-cpu Nehalem "*"/lilbee_engine_tei/bin/text-embeddings-router --version" ]]
}

@test "a CPU router serves a model on this CPU and on the emulated CPU" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(cut -d' ' -f2 "${ROUTER_LOG}" | tr '\n' ' ')" = "--version --model-id --version --model-id " ]
  [ "$(wc -l < "${QEMU_LOG}")" -eq 2 ]
  [ "$(grep -c '^embedding length 384$' <<< "$output")" -eq 2 ]
}

@test "the first start ignores an emulated CPU of the shell" {
  cuda
  EMULATE_CPU=other run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(wc -l < "${QEMU_LOG}")" -eq 1 ]
  grep -q '^-cpu Nehalem ' "${QEMU_LOG}"
}

@test "a CUDA router finds the driver stub of the build" {
  cuda
  tool ldd 'echo "${LD_LIBRARY_PATH-}" >> "${STEP_LOG}"'
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(cut -d: -f1 "${STEP_LOG}")" = "${ROOT}/out/stubs
${ROOT}/out/stubs" ]
}

@test "the modes of the wheel project do not come from the checkout" {
  chmod 600 "${TREE}/lilbee_engine_tei/__init__.py"
  chmod 700 "${TREE}/lilbee_engine_tei"
  umask 0077
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(find "${TREE}/lilbee_engine_tei/__init__.py" -perm 644 | wc -l)" -eq 1 ]
  [ "$(find "${TREE}" -type d ! -perm 755 | wc -l)" -eq 0 ]
  [ "$(find "${ROOT}/out" -maxdepth 0 -perm 755 | wc -l)" -eq 1 ]
}

@test "the Metal build compiles the candle and metal backends for the first Apple CPU" {
  metal
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(grep -c '^cargo build' "${STEP_LOG}")" -eq 1 ]
  grep -qF "cargo build --release --locked --bin text-embeddings-router --no-default-features --features candle,metal,http | flags=-C target-cpu=apple-m1 " "${STEP_LOG}"
  grep -qx 'rustup toolchain install 1.92.0 --profile minimal' "${STEP_LOG}"
  grep -q '^cargo build .* toolchain=1.92.0 .* floor=14.0 ' "${STEP_LOG}"
  [[ "$output" == *"Xcode 26.0"* ]]
  [ "$(ls "${OUT}")" = "$(printf '%s\n' license-expression.txt "${WHEEL}" "${WHEEL}.sha256" | sort)" ]
}

@test "the Metal build uses one folder for a tag and a cargo home of its own" {
  metal
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  grep -q "^cargo build .* home=${WORK}/cargo .* cwd=${SRC}$" "${STEP_LOG}"
  grep -q "^cargo-about | home=${WORK}/cargo$" "${STEP_LOG}"
}

@test "a cargo variable of the shell does not reach the Metal build" {
  metal
  CARGO_TARGET_DIR=/x CARGO_ENCODED_RUSTFLAGS=-Cx CARGO_HOME=/y RUSTC_WRAPPER=sccache RUSTC=/r RUSTC_WORKSPACE_WRAPPER=/w RUSTFLAGS=-Cz run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  grep -q "flags=-C target-cpu=apple-m1 .* home=${WORK}/cargo target=unset encoded=unset wrapper=unset rustc=unset workspace=unset " "${STEP_LOG}"
}

@test "the Metal build compiles the source of its tag" {
  metal
  local tag
  for tag in v1.9.4 v1.9.5; do
    mkdir -p "${ROOT}-${tag}"
    cp -R "${ROOT}/." "${ROOT}-${tag}/"
    TEI_TAG="${tag}" run bash "${ROOT}-${tag}/tools/wheel-build/pack_tei.sh"
    [ "$status" -eq 0 ]
    [ "$(git -C "${TMPDIR}lilbee-tei-${tag}/src" rev-parse HEAD)" = "$(git -C "${TEI_REPO}" rev-parse "${tag}")" ]
  done
}

@test "a router of another compiler than the pinned Rust stops the Metal build" {
  metal
  local commits copy
  # Another cargo, two compilers, and a router without the commit of a compiler.
  for commits in 1100 "192 1100" ""; do
    copy="${ROOT}-${commits// /-}"
    mkdir -p "${copy}" "${copy}-tmp"
    cp -R "${ROOT}/." "${copy}/"
    TMPDIR="${copy}-tmp" COMPILERS="${commits}" run bash "${copy}/tools/wheel-build/pack_tei.sh"
    [ "$status" -eq 1 ]
    [[ "$output" == *"pack_tei: the router comes from "*", not from Rust 1.92.0 "* ]]
  done
  [ "$(grep -c '^cargo build ' "${STEP_LOG}")" -eq 3 ]
  run grep '^wheel ' "${STEP_LOG}"
  [ "$status" -eq 1 ]
  [ ! -e "${OUT}" ]
}

@test "a folder of an earlier Metal build is refused" {
  metal
  mkdir -p "${WORK}"
  run bash "${SCRIPT}"
  [ "$status" -eq 2 ]
  [[ "$output" == *"${WORK} exists"* ]]
  [ ! -e "${STEP_LOG}" ]
}

@test "a tag with another Rust toolchain stops the Metal build" {
  metal
  RUSTC_VERSION=1.91.0 run bash "${SCRIPT}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"does not name Rust 1.91.0"* ]]
  run grep '^cargo build' "${STEP_LOG}"
  [ "$status" -eq 1 ]
}

# Make curl write a cargo-about release that another build made. Its cargo-about appends to STEP_LOG.
other_cargo_about() {  # triple
  local name="cargo-about-0.8.4-$1"
  mkdir -p "release/${name}"
  printf '%s\n' '#!/bin/sh' 'echo "other cargo-about" >> "${STEP_LOG}"' > "release/${name}/cargo-about"
  chmod +x "release/${name}/cargo-about"
  tar -czf release.tar.gz -C release "${name}"
  export OTHER_RELEASE="${PWD}/release.tar.gz"
  tool curl 'while [ "$#" -gt 1 ]; do [ "$1" != -o ] || cp "${OTHER_RELEASE}" "$2"; shift; done'
}

@test "a cargo-about digest that differs stops the Metal build before cargo-about runs" {
  metal
  cp "${REPO_ROOT}/tools/wheel-build/install_cargo_about.sh" "${ROOT}/tools/wheel-build/"
  other_cargo_about aarch64-apple-darwin
  UNAME_M=arm64 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"install_cargo_about: digest "* ]]
  run grep -E '^(other cargo-about|cargo build)' "${STEP_LOG}"
  [ "$status" -eq 1 ]
  [ ! -e "${OUT}" ]
}

@test "a cargo-about digest that differs stops the install in the Linux image before cargo-about runs" {
  other_cargo_about x86_64-unknown-linux-musl
  run bash "${REPO_ROOT}/tools/wheel-build/install_cargo_about.sh" "${BATS_TEST_TMPDIR}/about"
  [ "$status" -ne 0 ]
  [[ "$output" == *"install_cargo_about: digest "* ]]
  [ ! -e "${STEP_LOG}" ]
  [ ! -e "${BATS_TEST_TMPDIR}/about" ]
}

@test "a Metal compile that fails stops the pack" {
  metal
  COMPILE_FAIL=1 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [ ! -e "${OUT}" ]
}

@test "the Metal wheel comes from the router of the build and from the pinned tools" {
  metal
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  grep -qxF "wheel candle,metal,http aarch64-apple-darwin metal macosx_14_0_arm64 out | src=${SRC} router=${SRC}/target/release/text-embeddings-router python=python locale=C path=${WORK}/bin" "${STEP_LOG}"
  grep -qxF -- '-m pip install -q build wheel delocate | constraint=../../tools/wheel-build/build-tools.constraints ../../tools/wheel-build/tei-tools.constraints cwd=engine-wheel-tei' "${PYTHON_LOG}"
}

@test "the install of the Metal wheel reads no constraint" {
  metal
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  grep -q '^-m pip install -q wheel/.* | constraint=unset ' "${PYTHON_LOG}"
}

@test "delocate runs after the wheel build with the commit time of the source and the macOS floor" {
  metal
  SOURCE_DATE_EPOCH=1 run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(sed -E 's/[ -].*//' "${STEP_LOG}" | tr '\n' ' ')" = "rustup cargo cargo wheel delocate " ]
  grep -qxF "delocate --require-archs arm64 --require-target-macos-version 14.0 -w out/repaired out/dist/${WHEEL} | epoch=1700000000" "${STEP_LOG}"
}

@test "a binary above the macOS floor stops the pack" {
  metal
  DELOCATE_FAIL=1 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [ ! -e "${OUT}" ]
}

@test "a library that delocate grafts stops the pack" {
  metal
  GRAFT=1 run bash "${SCRIPT}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"delocate grafted a library"* ]]
  [ ! -e "${OUT}" ]
}

@test "a Metal router starts once, without the emulator" {
  metal
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(cat "${ROUTER_LOG}")" = "router --version" ]
  [ ! -e "${QEMU_LOG}" ]
}

@test "a Metal router that does not start is refused" {
  metal
  ROUTER_FAIL=1 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [ ! -e "${OUT}" ]
}

@test "the constraints pin every tool of the two builds to one version and one digest" {
  local dir="${REPO_ROOT}/tools/wheel-build" name
  for name in altgraph cmake delocate macholib ninja typing_extensions; do
    grep -qE "^${name}==[0-9.]+ \\\\$" "${dir}/tei-tools.constraints"
  done
  grep -qE '^delocate==0\.13\.0 ' "${dir}/tei-tools.constraints"
  [ "$(grep -cE '^[a-z_]+==' "${dir}/tei-tools.constraints")" -eq 6 ]
  [ "$(grep -cE '^    --hash=sha256:[0-9a-f]{64}$' "${dir}/tei-tools.constraints")" -eq 6 ]
  # Every package that tei_manylinux.sh installs with pip has a pin.
  for name in $(sed -n 's/.* -m pip install -q //p' "${dir}/tei_manylinux.sh"); do
    grep -qE "^${name}==" "${dir}/tei-tools.constraints" "${dir}/build-tools.constraints"
  done
}
