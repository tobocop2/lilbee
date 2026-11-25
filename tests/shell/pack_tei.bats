#!/usr/bin/env bats
# tools/wheel-build/pack_tei.sh in a copy of the tree, with stubs for docker and the Rust tools. The venvs and their pip
# are real. pip installs stand-ins for the build tools from a folder, by the names of the real requirements files. The
# wheel build script, the wheel check and the smoke test are the real ones, on a wheel whose router is a shell script.
# tests/test_tei_wheel_tools.py tests tei_manylinux.sh, tei_wheel.sh, check_tei_wheel.py and smoke_tei.sh.

# Make one venv that the tests copy, of a Python 3.12 with the pip that the Python bundles. A computer without such a
# Python fails the tests.
setup_file() {
  local base
  export VENV_TEMPLATE="${BATS_FILE_TMPDIR}/venv"
  for base in python3.12 python3 python "$(uv python find 3.12 2> /dev/null || true)"; do
    [ -n "${base}" ] || continue
    [ "$("${base}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2> /dev/null)" = 3.12 ] || continue
    "${base}" -m venv --clear "${VENV_TEMPLATE}" > /dev/null 2>&1 || continue
    REAL_PYTHON="$(command -v "${base}")"
    export REAL_PYTHON
    return
  done
  echo "no Python 3.12 here makes a venv with pip: run 'uv python install 3.12'" >&2
  return 1
}

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

  cp "${STUBS}/python-tei-pack" "${BIN}/python"
  cp "${STUBS}/docker-tei-pack" "${BIN}/docker"
  cp "${STUBS}/tei-wheel-pack" "${STUBS}/uname" "${BIN}/"
  # The real qemu and ldd do not read a shell script.
  tool qemu-x86_64-static 'echo "$*" >> "${QEMU_LOG}"' '[ -z "${QEMU_FAIL-}" ] || exit 132' 'shift 2' 'exec "$@"'
  tool ldd 'echo "${LDD_TEXT:-libc.so.6 => /lib/libc.so.6}"'
  tool curl 'case "$*" in */embed*) echo "[[$(yes 0 | head -n 384 | paste -s -d, -)]]" ;; esac'
  PATH="${BIN}:${PATH}"

  export DOCKER_LOG="${BATS_TEST_TMPDIR}/docker.log" PYTHON_LOG="${BATS_TEST_TMPDIR}/python.log" PYTHON_CALLS="${BATS_TEST_TMPDIR}/calls.log"
  export QEMU_LOG="${BATS_TEST_TMPDIR}/qemu.log" ROUTER_LOG="${BATS_TEST_TMPDIR}/router.log" ROUTER_FAIL=
  export STEP_LOG="${BATS_TEST_TMPDIR}/step.log" TMPDIR="${BATS_TEST_TMPDIR}/tmp/"
  export TEI_REPO="${BATS_TEST_TMPDIR}/tei-origin" TEI_TAG=v1.9.4 RUSTC_VERSION=1.92.0 WHEEL_VERSION=1.9.4 BUILD_NUMBER=2
  export CUDA_VERSION=12.9.1 CUDA_PACKAGE_SUFFIX=12-9 CUDA_CC= BACKEND=cpu PLATFORM=manylinux_2_34_x86_64
  export MANYLINUX_IMAGE=quay.io/pypa/x@sha256:abc RUNNER_OS=Linux OUT="${BATS_TEST_TMPDIR}/packs-out"
  export PIP_NO_INDEX=1 PIP_FIND_LINKS="${BATS_TEST_TMPDIR}/wheels" PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_CONFIG_FILE=/dev/null
  unset SOURCE_DATE_EPOCH EMULATE_CPU STUB_DIR PIP_CONSTRAINT PIP_REQUIRE_HASHES CARGO_HOME RUSTFLAGS
  WHEEL=lilbee_engine_tei-1.9.4-2.cpu-py3-none-manylinux_2_34_x86_64.whl
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
  WHEEL=lilbee_engine_tei-1.9.4-2.cuda_sm80-py3-none-manylinux_2_34_x86_64.whl
}

# Set up the macOS cell. The Rust and Mac tools append to STEP_LOG, in the order of the calls.
metal() {
  export BACKEND=metal PLATFORM=macosx_14_0_arm64 MACOSX_DEPLOYMENT_TARGET=14.0 RUNNER_OS=macOS UNAME_S=Darwin
  WHEEL=lilbee_engine_tei-1.9.4-2.metal-py3-none-macosx_14_0_arm64.whl
  WORK="${BATS_TEST_TMPDIR}/tmp/lilbee-tei-v1.9.4"
  SRC="${WORK}/src"
  tool otool 'echo /usr/lib/libSystem.B.dylib'
  tool xcodebuild 'echo "Xcode 26.0"'
  tool rustup 'if [ "$*" = "run 1.92.0 rustc -vV" ]; then printf "rustc 1.92.0\ncommit-hash: %040d\nrelease: 1.92.0\n" 192; exit; fi' \
    'echo "rustup $*" >> "${STEP_LOG}"'
  # The router holds the commit of each compiler in COMPILERS, as the routers of rustc do.
  # cargo-about writes the file that -o names.
  tool cargo 'if [ "$1" = about ]; then text=licences; while [ "$#" -gt 0 ]; do case "$1" in --format) text="{\"licenses\":[{\"id\":\"MIT\"}]}" ;; -o) out="$2" ;; esac; shift; done; echo "${text}" > "${out}"; exit; fi' \
    'mkdir -p "${CARGO_HOME}/registry/src/index-1/kernels-0.1.0"' 'echo notice > "${CARGO_HOME}/registry/src/index-1/kernels-0.1.0/NOTICE"' 'echo "cargo $* | flags=${RUSTFLAGS-} toolchain=${RUSTUP_TOOLCHAIN-} home=${CARGO_HOME-} target=${CARGO_TARGET_DIR-unset} encoded=${CARGO_ENCODED_RUSTFLAGS-unset} wrapper=${RUSTC_WRAPPER-unset} rustc=${RUSTC-unset} workspace=${RUSTC_WORKSPACE_WRAPPER-unset} floor=${MACOSX_DEPLOYMENT_TARGET-} cwd=${PWD}" >> "${STEP_LOG}"' \
    '[ -z "${COMPILE_FAIL-}" ] || exit 101' 'mkdir -p target/release' 'exec > target/release/text-embeddings-router' 'printf "router\0"' \
    'for commit in ${COMPILERS-192 192}; do printf "/rustc/%040d/library/core\0" "${commit}"; done'
  # The stand-in of delocate that pip installs runs this one.
  tool stub-delocate-wheel 'echo "delocate $* | epoch=${SOURCE_DATE_EPOCH-unset}" >> "${STEP_LOG}"' '[ -z "${DELOCATE_FAIL-}" ] || exit 1' \
    'mkdir -p out/repaired' 'cp out/dist/*.whl out/repaired/' \
    '[ -z "${GRAFT-}" ] || (cd out/repaired && "${REAL_PYTHON}" -c "import glob, zipfile; zipfile.ZipFile(glob.glob(\"*.whl\")[0], \"a\").writestr(\"lilbee_engine_tei/.dylibs/libx.dylib\", \"x\")")'
  printf '%s\n' 'echo "cargo-about | home=${CARGO_HOME-}" >> "${STEP_LOG}"' 'echo about > "$1/cargo-about"' > "${ROOT}/tools/wheel-build/install_cargo_about.sh"
  # The real script, after a line in the log.
  mv "${ROOT}/tools/wheel-build/tei_wheel.sh" "${ROOT}/tools/wheel-build/real_tei_wheel.sh"
  printf '%s\n' 'test -s "$2"' 'test "$(cat "${1%/src}/bin/cargo-about")" = about' \
    'echo "wheel ${*:3} | src=$1 router=$2 python=${PYTHON} locale=${LC_ALL-} path=${PATH%%:*}" >> "${STEP_LOG}"' \
    'exec bash "$(dirname "$0")/real_tei_wheel.sh" "$@"' > "${ROOT}/tools/wheel-build/tei_wheel.sh"
  "${REAL_PYTHON}" "${BATS_TEST_DIRNAME}/tei_build_tools.py" "${PIP_FIND_LINKS}" \
    "${ROOT}/tools/wheel-build/build-tools.constraints" "${ROOT}/tools/wheel-build/tei-tools-macos.constraints"
  mkdir -p "${TEI_REPO}"
  printf '[toolchain]\nchannel = "1.92.0"\n' > "${TEI_REPO}/rust-toolchain.toml"
  echo licence > "${TEI_REPO}/LICENSE"
  echo 'version = 4' > "${TEI_REPO}/Cargo.lock"
  git -C "${TEI_REPO}" init -q -b main
  git -C "${TEI_REPO}" add rust-toolchain.toml LICENSE Cargo.lock
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
  for name in TEI_REPO TEI_TAG RUSTC_VERSION WHEEL_VERSION BUILD_NUMBER BACKEND PLATFORM OUT CUDA_VERSION CUDA_PACKAGE_SUFFIX MANYLINUX_IMAGE; do
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
  [ ! -e "${PYTHON_LOG}" ]
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
  for name in TEI_REPO TEI_TAG RUSTC_VERSION WHEEL_VERSION BUILD_NUMBER BACKEND CUDA_CC CUDA_PACKAGE_SUFFIX CUDA_VERSION; do
    want="${want}
-e
${name}"
  done
  want="${want}
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
    lilbee_engine_tei-1.9.4-2.cpu-py3-none-linux_x86_64.whl \
    lilbee_engine_tei-1.9.4-2.cpu-py3-none-manylinux_2_34_x86_64.manylinux_2_28_x86_64.whl \
    lilbee_engine_tei-1.9.4-2.metal-py3-none-manylinux_2_34_x86_64.whl \
    lilbee_engine_tei-1.9.5-2.cpu-py3-none-manylinux_2_34_x86_64.whl \
    lilbee_engine_tei-1.9.4-1.cpu-py3-none-manylinux_2_34_x86_64.whl \
    lilbee_engine_tei-1.9.4-2.cpu-cp312-none-manylinux_2_34_x86_64.whl; do
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
  VENV_TEMPLATE= run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ -x "${ROOT}/pack-venv/bin/python" ]
  [ "$(grep -c '^-m venv pack-venv$' "${PYTHON_LOG}")" -eq 1 ]
}

# Pass when each Python call after the venv of the checkout runs an interpreter of a venv, and each call in "$@" is one.
only_venv_python() {  # call...
  local call
  sed '1,/ -m venv pack-venv$/d' "${PYTHON_CALLS}" > calls.after
  for call in "$@"; do grep -qF -- "${ROOT}/pack-venv/bin/python ${call}" calls.after; done
  run grep -vF -e "${ROOT}/pack-venv/bin/python " -e "${ROOT}/pack-venv/bin/python3 " -e 'check-venv/bin/python ' calls.after
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
  only_venv_python '-m pip install --require-hashes --force-reinstall -r ' '-m build --wheel --no-isolation ' '-m wheel tags ' '-m zipfile -l ' 'tools/wheel-build/check_tei_wheel.py ' '-m venv check-venv'
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

@test "the Metal wheel comes from the router of the build and builds without isolation" {
  metal
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  grep -qxF "wheel candle,metal,http aarch64-apple-darwin metal macosx_14_0_arm64 out | src=${SRC} router=${SRC}/target/release/text-embeddings-router python=python locale=C path=${WORK}/bin" "${STEP_LOG}"
  grep -qxF 'build --wheel --no-isolation -C--build-option=--build-number=2.metal | cwd=engine-wheel-tei' "${STEP_LOG}"
}

@test "the Metal build installs each tool of its two files, and no other, with the pip that the Python bundles" {
  metal
  local want site
  want="$(sed -n 's/^\([a-z_]*\)==.*/\1-99.dist-info/p' "${REPO_ROOT}"/tools/wheel-build/{build-tools,tei-tools-macos}.constraints | sort)"
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  grep -qF -- '-m pip install --require-hashes --force-reinstall -r tools/wheel-build/build-tools.constraints -r tools/wheel-build/tei-tools-macos.constraints | ' "${PYTHON_LOG}"
  site="$(echo "${ROOT}"/pack-venv/lib/python3*/site-packages)"
  [ "$(find "${site}" -maxdepth 1 -name '*-99.dist-info' -exec basename {} \; | sort)" = "${want}" ]
}

@test "the variables of the Metal build do not reach the checks" {
  metal
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  grep -q '^-m pip install -q wheel/.* | constraint=unset flags=unset ' "${PYTHON_LOG}"
}

# Pass when the Metal build stops at pip, before the wheel build.
stops_at_pip() {  # text
  [ "$status" -eq 1 ]
  [[ "$output" == *"$1"* ]]
  grep -q '^cargo build ' "${STEP_LOG}"
  run grep '^wheel ' "${STEP_LOG}"
  [ "$status" -eq 1 ]
  [ ! -e "${OUT}" ]
}

@test "a build tool with another digest stops the Metal build" {
  metal
  local name
  for name in setuptools:build-tools delocate:tei-tools-macos; do
    cp -R "${ROOT}" "${ROOT}-${name%:*}"
    sed "s/^${name%:*}==99 .*/${name%:*}==99 --hash=sha256:$(printf '%064d' 0)/" \
      "${ROOT}/tools/wheel-build/${name#*:}.constraints" > "${ROOT}-${name%:*}/tools/wheel-build/${name#*:}.constraints"
    mkdir "${BATS_TEST_TMPDIR}/tmp-${name%:*}"
    TMPDIR="${BATS_TEST_TMPDIR}/tmp-${name%:*}" run bash "${ROOT}-${name%:*}/tools/wheel-build/pack_tei.sh"
    stops_at_pip "DO NOT MATCH THE HASHES"
  done
}

@test "a tool that needs one which no file lists stops the Metal build" {
  metal
  local name
  for name in packaging:build-tools altgraph:tei-tools-macos; do
    cp -R "${ROOT}" "${ROOT}-${name%:*}"
    grep -v "^${name%:*}==" "${ROOT}/tools/wheel-build/${name#*:}.constraints" > "${ROOT}-${name%:*}/tools/wheel-build/${name#*:}.constraints"
    mkdir "${BATS_TEST_TMPDIR}/tmp-${name%:*}"
    TMPDIR="${BATS_TEST_TMPDIR}/tmp-${name%:*}" run bash "${ROOT}-${name%:*}/tools/wheel-build/pack_tei.sh"
    stops_at_pip "In --require-hashes mode, all requirements must have their versions pinned with =="
  done
}

@test "delocate runs after the wheel build with the commit time of the source and the macOS floor" {
  metal
  SOURCE_DATE_EPOCH=1 run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(sed -E 's/[ -].*//' "${STEP_LOG}" | tr '\n' ' ')" = "rustup cargo cargo wheel build delocate " ]
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

# Print the tools of a requirements file in one line. A pair of lines that is not one exact version and its digest
# prints those pairs, and no tools: the shell of "$(...)" does not stop at a failed check.
pinned_tools() {  # file
  grep -v '^#' "$1" | paste - - > pairs
  if grep -vE $'^[a-z_]+==[0-9.]+ \\\\\t    --hash=sha256:[0-9a-f]{64}$' pairs; then return 1; fi
  sed 's/==.*//' pairs | paste -s -d ' ' -
}

@test "a tool without an exact version or its digest hides the tools of its file" {
  local digest last
  digest="    --hash=sha256:$(printf '%064d' 0)"
  printf '%s\n' '# tools' 'a==1.2 \' "${digest}" 'b_c==3 \' "${digest}" > exact
  [ "$(pinned_tools exact)" = "a b_c" ]
  for last in 'b_c==3' 'b_c==3 \' $'b_c==3\n'"${digest}" $'b_c==3 \\\n'"${digest}0" $'b_c==3 \\\n    --hash=sha256:0' \
    $'b_c==3.* \\\n'"${digest}"; do
    printf '%s\n' 'a==1.2 \' "${digest}" "${last}" > loose
    [ "$(pinned_tools loose)" != "a b_c" ] || { echo "not refused: ${last}"; return 1; }
  done
}

# pip checks the digest of the first file only when two files of one call name a tool.
@test "the two files of an install name no tool twice" {
  local dir="${REPO_ROOT}/tools/wheel-build" file names
  for file in tei-tools-linux tei-tools-macos; do
    names="$(pinned_tools "${dir}/build-tools.constraints") $(pinned_tools "${dir}/${file}.constraints")"
    [ "$(tr ' ' '\n' <<< "${names}" | sort | uniq -d)" = "" ]
  done
}

@test "each file pins the tools of its one build to one version and one digest" {
  local dir="${REPO_ROOT}/tools/wheel-build"
  [ "$(pinned_tools "${dir}/tei-tools-linux.constraints")" = "cmake ninja" ]
  [ "$(pinned_tools "${dir}/tei-tools-macos.constraints")" = "altgraph delocate macholib typing_extensions" ]
  grep -qxF 'delocate==0.13.0 \' "${dir}/tei-tools-macos.constraints"
}
