#!/usr/bin/env bats
# tools/wheel-build/pack_mlx.sh in a copy of the tree, with a stand-in for mlx_wheel.sh that writes a small wheel and a
# stand-in for the Python of the build venv. The tag check, the licence check and the start of the chat server are the
# real ones, on a wheel whose server is a shell script. tests/test_engine_mlx_pin.py tests mlx_wheel.sh.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  ROOT="${BATS_TEST_TMPDIR}/lil bee"
  BIN="${BATS_TEST_TMPDIR}/bin"
  mkdir -p "${ROOT}/tools" "${ROOT}/engines" "${BIN}"
  cp -R "${REPO_ROOT}/tools/wheel-build" "${ROOT}/tools/"
  cp "${REPO_ROOT}/engines/mlx.env" "${ROOT}/engines/"
  cp "${BATS_TEST_DIRNAME}/stubs/mlx-wheel-pack" "${ROOT}/tools/wheel-build/mlx_wheel.sh"
  SCRIPT="${ROOT}/tools/wheel-build/pack_mlx.sh"

  REAL_PYTHON="$(command -v python3)"
  cp "${BATS_TEST_DIRNAME}/stubs/python-mlx-pack" "${BIN}/python"
  printf '%s\n' '#!/bin/sh' 'exit 0' > "${BIN}/zstd"
  chmod +x "${BIN}"/*
  PATH="${BIN}:${PATH}"

  export REAL_PYTHON PYTHON_LOG="${BATS_TEST_TMPDIR}/python.log"
  export MLX_BUILD_LOG="${BATS_TEST_TMPDIR}/build.log" MLX_SERVER_LOG="${BATS_TEST_TMPDIR}/server.log"
  export BACKEND=metal PLATFORM=macosx_14_0_arm64 BUILD_NUMBER=1 LICENSE_TERMS=MIT
  export RUNNER_OS=macOS OUT="${BATS_TEST_TMPDIR}/packs-out"
  WHEEL=lilbee_engine_mlx-0.32.2-1.metal-py3-none-macosx_14_0_arm64.whl
  cd "${BATS_TEST_TMPDIR}"
}

@test "the wheel, its sha256 file and the licence expression reach the output folder" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(ls "${OUT}")" = "${WHEEL}
${WHEEL}.sha256
license-expression.txt" ]
  (cd "${OUT}" && sha256sum -c "${WHEEL}.sha256")
  [ "$(cat "${OUT}/license-expression.txt")" = "MIT AND Apache-2.0" ]
}

@test "the expression of the wheel is the expression the build wrote, not only the cell terms" {
  MLX_EXPRESSION="MIT AND PSF-2.0 AND MPL-2.0" run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(cat "${OUT}/license-expression.txt")" = "MIT AND PSF-2.0 AND MPL-2.0" ]
}

@test "every variable of the contract is required" {
  local name
  for name in BACKEND PLATFORM BUILD_NUMBER LICENSE_TERMS OUT; do
    run env -u "${name}" bash "${SCRIPT}"
    [ "$status" -eq 2 ]
    [[ "$output" == *"${name} is not set"* ]]
  done
  [ ! -e "${MLX_BUILD_LOG}" ]
}

@test "a computer that is not a Mac builds no cell" {
  RUNNER_OS=Linux run bash "${SCRIPT}"
  [ "$status" -eq 2 ]
  [[ "$output" == *"builds on macOS only"* ]]
  [ ! -e "${MLX_BUILD_LOG}" ]
}

@test "a Mac without zstd is refused before the build" {
  PATH="${BIN}:/usr/bin:/bin"
  mv "${BIN}/zstd" "${BIN}/zstd-gone"
  if command -v zstd > /dev/null; then skip "this computer has zstd in /usr/bin or /bin"; fi
  run bash "${SCRIPT}"
  [ "$status" -eq 2 ]
  [[ "$output" == *"zstd is not installed"* ]]
  [ ! -e "${MLX_BUILD_LOG}" ]
}

@test "a backend that the pin file does not name is refused before the build" {
  BACKEND=cpu run bash "${SCRIPT}"
  [ "$status" -eq 2 ]
  [[ "$output" == *"names the backend cpu"*"names metal"* ]]
  [ ! -e "${MLX_BUILD_LOG}" ]
}

@test "a platform that the pin file does not name is refused before the build" {
  PLATFORM=macosx_15_0_arm64 run bash "${SCRIPT}"
  [ "$status" -eq 2 ]
  [[ "$output" == *"names the platform macosx_15_0_arm64"*"names macosx_14_0_arm64"* ]]
  [ ! -e "${MLX_BUILD_LOG}" ]
}

@test "each folder that an earlier build leaves is refused" {
  local name
  for name in out wheel pack-venv check; do
    mkdir "${ROOT}/${name}"
    run bash "${SCRIPT}"
    [ "$status" -eq 2 ]
    [[ "$output" == *"(${name} exists)"* ]]
    mv "${ROOT}/${name}" "${BATS_TEST_TMPDIR}/moved-${name}"
  done
  [ ! -e "${MLX_BUILD_LOG}" ]
}

@test "a Python other than 3.12 is refused before the build" {
  local name
  for name in python3.12 python3; do cp "${BIN}/python" "${BIN}/${name}"; done
  PYTHON_VERSION=3.11 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"Python 3.12 is not on PATH"* ]]
  [ ! -e "${MLX_BUILD_LOG}" ]
}

@test "pip installs the pinned build tools and the mlx tools by their digests in the build venv" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(wc -l < "${PYTHON_LOG}")" -eq 1 ]
  grep -qF -- "-r tools/wheel-build/build-tools.constraints -r tools/wheel-build/mlx-tools-macos.constraints" "${PYTHON_LOG}"
  grep -qF " exe=${ROOT}/pack-venv/bin/python" "${PYTHON_LOG}"
}

@test "the build gets the build number and the uv and delocate-wheel of the build venv" {
  BUILD_NUMBER=7 MLX_WHEEL_NAME=lilbee_engine_mlx-0.32.2-7.metal-py3-none-macosx_14_0_arm64.whl run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  grep -qx "args=out build=7" "${MLX_BUILD_LOG}"
  grep -qx "uv=${ROOT}/pack-venv/bin/uv delocate=${ROOT}/pack-venv/bin/delocate-wheel" "${MLX_BUILD_LOG}"
}

@test "the mlx tools pin uv and delocate at the versions of the pin file" {
  local file="${REPO_ROOT}/tools/wheel-build/mlx-tools-macos.constraints"
  grep -qx "uv==$(sed -n 's/^MLX_UV_VERSION=//p' "${REPO_ROOT}/engines/mlx.env") \\\\" "${file}"
  grep -qx "delocate==$(sed -n 's/^MLX_DELOCATE_VERSION=//p' "${REPO_ROOT}/engines/mlx.env") \\\\" "${file}"
  [ "$(grep -c '^[[:alnum:]].*==' "${file}")" -eq "$(grep -c -- '--hash=sha256:' "${file}")" ]
}

@test "a build that fails stops the pack" {
  MLX_BUILD_FAIL=1 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [ ! -e "${OUT}" ]
}

@test "a wheel with another build number in its name is refused" {
  MLX_WHEEL_NAME=lilbee_engine_mlx-0.32.2-2.metal-py3-none-macosx_14_0_arm64.whl run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"not ${WHEEL}"* ]]
  [ ! -e "${OUT}" ]
}

@test "a wheel with a wrong tag in its WHEEL file is refused" {
  MLX_WHEEL_TAG=py3-none-macosx_11_0_arm64 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"does not carry macosx_14_0_arm64"* ]]
  [ ! -e "${OUT}" ]
}

@test "a licence expression that does not start with the cell terms is refused" {
  MLX_EXPRESSION="Apache-2.0 AND MIT" run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"does not start with the cell terms 'MIT'"* ]]
  [ ! -e "${OUT}" ]
}

@test "a cell term that is only the start of a longer term is refused" {
  MLX_EXPRESSION="MIT-0 AND Apache-2.0" run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [ ! -e "${OUT}" ]
}

@test "the chat server of the unpacked wheel starts with --help" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(cat "${MLX_SERVER_LOG}")" = "--help" ]
}

@test "a chat server that does not start is refused" {
  MLX_SERVER_FAIL=1 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"chat server of the wheel does not start"* ]]
  [ ! -e "${OUT}" ]
}
