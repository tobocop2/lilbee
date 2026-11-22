#!/usr/bin/env bats
# tools/wheel-build/pack_ik.sh in a copy of the tree, with stubs for docker, ldd, objdump and the Python build tools.
# The licence collector, the project stamp, the wheel check and the glibc check are the real ones, on a source tree
# and a server that the docker stub writes. tests/test_ik_llama_adapter.py tests check_ik_wheel.py and
# ik_llama_manylinux.sh.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  STUBS="${BATS_TEST_DIRNAME}/stubs"
  ROOT="${BATS_TEST_TMPDIR}/lil bee"
  BIN="${BATS_TEST_TMPDIR}/bin"
  ADAPTER=src/lilbee/providers/fleet/engines/ik_llama.py
  mkdir -p "${ROOT}/packaging" "${ROOT}/tools" "${ROOT}/${ADAPTER%/*}" "${BIN}"
  cp -R "${REPO_ROOT}/tools/wheel-build" "${ROOT}/tools/"
  cp -R "${REPO_ROOT}/packaging/engine-wheel-ik-llama-cpp" "${ROOT}/packaging/"
  cp "${REPO_ROOT}/${ADAPTER}" "${ROOT}/${ADAPTER}"
  SCRIPT="${ROOT}/tools/wheel-build/pack_ik.sh"
  TREE="${ROOT}/packaging/engine-wheel-ik-llama-cpp"

  REAL_PYTHON="$(command -v python3)"
  cp "${STUBS}/python-ik-pack" "${BIN}/python"
  cp "${STUBS}/docker-ik-pack" "${BIN}/docker"
  # The real ldd and objdump do not read a shell script.
  tool ldd 'echo "ldd ${1##*/} | libraries=${LD_LIBRARY_PATH-unset}" >> "${SERVER_LOG}"' 'echo "${LDD_TEXT:-libc.so.6 => /lib/libc.so.6}"' 'exit "${LDD_STATUS:-0}"'
  tool objdump 'echo "0000 DF *UND* 0000 (GLIBC_${GLIBC_NEEDS:-2.34}) memcpy"'
  PATH="${BIN}:${PATH}"

  export REAL_PYTHON DOCKER_LOG="${BATS_TEST_TMPDIR}/docker.log" PYTHON_CALLS="${BATS_TEST_TMPDIR}/calls.log" SERVER_LOG="${BATS_TEST_TMPDIR}/server.log"
  export UPSTREAM_REPO=ikawrakow/ik_llama.cpp UPSTREAM_COMMIT=13cffc006b82e03e4d89cbb8c7327f0255ae3d18
  export BACKEND=cpu PLATFORM=manylinux_2_34_x86_64 BUILD_NUMBER=4 LICENSE_TERMS=MIT
  export MANYLINUX_IMAGE=quay.io/pypa/x@sha256:abc RUNNER_OS=Linux OUT="${BATS_TEST_TMPDIR}/packs-out"
  unset SOURCE_DATE_EPOCH PIP_CONSTRAINT LD_LIBRARY_PATH LICENSE_EXPRESSION LC_ALL
  INFO=lilbee_engine_ik_llama_cpp-0+git13cffc006b82.dist-info
  WHEEL=lilbee_engine_ik_llama_cpp-0+git13cffc006b82-4.cpu-py3-none-manylinux_2_34_x86_64.whl
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
  export BACKEND=cuda BUILD_NUMBER=3 LICENSE_TERMS='MIT AND LicenseRef-NVIDIA-CUDA-EULA'
  WHEEL=lilbee_engine_ik_llama_cpp-0+git13cffc006b82-3.cuda-py3-none-manylinux_2_34_x86_64.whl
}

# Print one file of the wheel in the output folder.
wheel_file() {  # path
  "${REAL_PYTHON}" -c 'import sys, zipfile; sys.stdout.write(zipfile.ZipFile(sys.argv[1]).read(sys.argv[2]).decode())' "${OUT}/${WHEEL}" "$1"
}

# Run the pack in a copy of the tree, with one variable set.
run_copy() {  # folder variable=value
  mkdir -p "${ROOT}-$1"
  cp -R "${ROOT}/." "${ROOT}-$1/"
  run env "$2" bash "${ROOT}-$1/tools/wheel-build/pack_ik.sh"
}

@test "the wheel and its sha256 file reach the output folder" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(ls "${OUT}")" = "$(printf '%s\n' "${WHEEL}" "${WHEEL}.sha256" | sort)" ]
  (cd "${OUT}" && sha256sum -c "${WHEEL}.sha256")
}

@test "every variable of the contract is required" {
  local name
  for name in UPSTREAM_REPO UPSTREAM_COMMIT BACKEND PLATFORM BUILD_NUMBER LICENSE_TERMS MANYLINUX_IMAGE OUT; do
    run env "${name}=" bash "${SCRIPT}"
    [ "$status" -eq 2 ]
    [[ "$output" == *"${name} is not set"* ]]
  done
  [ ! -e "${DOCKER_LOG}" ]
}

@test "a computer without a tool of the checks is refused before the build" {
  local bare="${BATS_TEST_TMPDIR}/bare" name missing
  for missing in git docker ldd objdump; do
    mkdir -p "${bare}-${missing}"
    for name in bash dirname uname git docker ldd objdump; do
      [ "${name}" = "${missing}" ] || ln -s "$(command -v "${name}")" "${bare}-${missing}/${name}"
    done
    PATH="${bare}-${missing}" run "${bare}-${missing}/bash" "${SCRIPT}"
    [ "$status" -eq 2 ]
    [[ "$output" == *"${missing} is not installed"* ]]
  done
  [ ! -e "${DOCKER_LOG}" ]
}

@test "macOS and Windows build no cell" {
  local os
  for os in macOS Windows; do
    RUNNER_OS="${os}" run bash "${SCRIPT}"
    [ "$status" -eq 2 ]
    [[ "$output" == *"no cell builds on ${os}"* ]]
  done
}

@test "a checkout that an earlier build used is refused" {
  local name calls
  for name in ik-src cuda-libs.sha256 cuda-stub vendored-terms.txt wheel pack-venv check-venv \
    packaging/engine-wheel-ik-llama-cpp/lilbee_engine_ik_llama_cpp/bin packaging/engine-wheel-ik-llama-cpp/dist; do
    mkdir "${ROOT}/${name}"
    run bash "${SCRIPT}"
    rmdir "${ROOT}/${name}"
    [ "$status" -eq 2 ]
    [[ "$output" == *"an earlier build used this checkout (${name} exists)"* ]]
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

@test "a cell builds in the pinned image with the checkout at /work" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(cat "${DOCKER_LOG}")" = "== call
run
--rm
-v
${ROOT}:/work
-w
/work
-e
UPSTREAM_REPO
-e
UPSTREAM_COMMIT
-e
BACKEND
-e
BIN_DIR=/work/packaging/engine-wheel-ik-llama-cpp/lilbee_engine_ik_llama_cpp/bin
-e
SRC_DIR=/work/ik-src
-e
SHA_FILE=/work/cuda-libs.sha256
-e
NOTICE_FILE=/work/packaging/engine-wheel-ik-llama-cpp/NOTICE-NVIDIA-CUDA
-e
OWNER=$(id -u):$(id -g)
${MANYLINUX_IMAGE}
bash
/work/tools/wheel-build/ik_llama_manylinux.sh" ]
}

@test "each variable that the build script requires reaches the container" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  local name names
  names="$(grep -v '^#' "${REPO_ROOT}/tools/wheel-build/ik_llama_manylinux.sh" | grep -oE '\$\{[A-Z_]+:\?\}' | tr -d '${}:?')"
  [ "$(wc -w <<< "${names}")" -eq 8 ]
  for name in ${names}; do grep -qE "^${name}(=|$)" "${DOCKER_LOG}"; done
}

@test "a build that fails stops the pack" {
  DOCKER_FAIL=build run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [ ! -e "${OUT}" ]
}

@test "the wheel carries the licence file of the source it builds" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(wheel_file "${INFO}/licenses/LICENSE-ik_llama.cpp")" = "$(cat "${ROOT}/ik-src/LICENSE")" ]
  [[ "$(cat "${ROOT}/ik-src/LICENSE")" == *"The ggml authors"* ]]
}

@test "the wheel carries the notices of both pages, the sqlite headers and the vendored code, and no other" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(wheel_file "${INFO}/licenses/THIRD-PARTY/ik_llama.cpp/MANIFEST")" = "LICENSE-Apache-2.0
examples/server/public/index.html.gz.NOTICE
examples/server/public_llamacpp/index_llamacpp.html.gz.NOTICE
examples/server/sqlite_modern_cpp/License.txt
tools/ui/LICENSE
vendor/cpp-httplib/httplib.h.NOTICE
vendor/miniaudio/miniaudio.h.NOTICE
vendor/nlohmann/json.hpp.NOTICE
vendor/stb/stb_image.h.NOTICE
vendor/stb/stb_image_resize2.h.NOTICE" ]
}

@test "an empty licence file of the source stops the pack" {
  EMPTY_LICENSE=1 run bash "${SCRIPT}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"empty licence file"*"LICENSE-ik_llama.cpp"* ]]
  [ ! -e "${OUT}" ]
}

@test "a source without a licence that the server needs stops the pack" {
  SOURCE_WITHOUT=examples/server/sqlite_modern_cpp run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"examples/server/sqlite_modern_cpp does not exist"* ]]
  [ ! -e "${OUT}" ]
}

@test "a licence collector that fails after it wrote the terms stops the pack" {
  echo 'exit 7' >> "${ROOT}/tools/wheel-build/collect_vendored_licences.sh"
  run bash "${SCRIPT}"
  [ "$status" -eq 7 ]
  [ -s "${ROOT}/vendored-terms.txt" ]
  [ ! -e "${TREE}/dist" ]
  [ ! -e "${OUT}" ]
}

@test "the wheel states the terms of the cell, then the vendored terms, each one time" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [[ "$(wheel_file "${INFO}/METADATA")" == *"License-Expression: MIT AND Apache-2.0 AND MIT-0 AND Unlicense"* ]]
}

@test "the CUDA wheel states the NVIDIA term of its cell" {
  cuda
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [[ "$(wheel_file "${INFO}/METADATA")" == *"License-Expression: MIT AND LicenseRef-NVIDIA-CUDA-EULA AND Apache-2.0 AND MIT-0 AND Unlicense"* ]]
}

@test "the terms and the notices do not depend on the locale or on a value of the shell" {
  # A sort that uses a locale other than C puts the lines in another order.
  tool sort 'if [ "${LC_ALL-}" = C ]; then exec "'"$(command -v sort)"'" "$@"; else exec "'"$(command -v sort)"'" -r "$@"; fi'
  # LC_ALL stays unset here: bash exports a local of a name that the shell exports.
  LANG=en_US.UTF-8 LICENSE_EXPRESSION=other run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [[ "$(wheel_file "${INFO}/METADATA")" == *"License-Expression: MIT AND Apache-2.0 AND MIT-0 AND Unlicense"* ]]
  [ "$(wheel_file "${INFO}/licenses/THIRD-PARTY/ik_llama.cpp/MANIFEST" | head -n 1)" = LICENSE-Apache-2.0 ]
}

@test "a licence expression that the lilbee pin does not name is refused" {
  LICENSE_TERMS='MIT AND BSD-3-Clause' run bash "${SCRIPT}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"the lilbee pin names"* ]]
  [ ! -e "${OUT}" ]
  [ ! -e "${SERVER_LOG}" ]
}

@test "a commit that the lilbee pin does not name is refused" {
  UPSTREAM_COMMIT=0000000000000000000000000000000000000000 run bash "${SCRIPT}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"another ik_llama.cpp commit"* ]]
  [ ! -e "${OUT}" ]
}

@test "the wheel build and the retag use the commit time of the source and the pinned tools" {
  SOURCE_DATE_EPOCH=1 run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  local tail='| epoch=1700000000 constraint=unset locale=unset cwd=engine-wheel-ik-llama-cpp'
  [ "$(grep -F 'cwd=engine-wheel-ik-llama-cpp' "${PYTHON_CALLS}")" = "${ROOT}/pack-venv/bin/python -m pip install --require-hashes --force-reinstall -r ../../tools/wheel-build/build-tools.constraints ${tail}
${ROOT}/pack-venv/bin/python -m build --wheel --no-isolation -C--build-option=--build-number=4.cpu ${tail}
${ROOT}/pack-venv/bin/python -m wheel tags --remove --platform-tag manylinux_2_34_x86_64 dist/${WHEEL/manylinux_2_34_x86_64/any} ${tail}" ]
}

@test "an install of the build tools that fails stops the pack before the wheel build" {
  PIP_STATUS=5 run bash "${SCRIPT}"
  [ "$status" -eq 5 ]
  [ ! -e "${TREE}/dist" ]
  [ ! -e "${OUT}" ]
}

@test "a requirements file that pins no tool stops the pack before the wheel build" {
  echo '# no tool' > "${ROOT}/tools/wheel-build/build-tools.constraints"
  run bash "${SCRIPT}"
  [ "$status" -eq 2 ]
  [[ "$output" == *"build-tools.constraints pins no tool"* ]]
  [ ! -e "${TREE}/dist" ]
  [ ! -e "${OUT}" ]
}

# Run ik_llama_manylinux.sh of the copy with IMAGE in place of the root of the container, up to its first cmake call.
# cmake ends with status 9.
run_server_build() {
  local script="${ROOT}/tools/wheel-build/ik_llama_manylinux.sh"
  export IMAGE="${BATS_TEST_TMPDIR}/image" STEPS="${BATS_TEST_TMPDIR}/steps.log"
  mkdir -p "${IMAGE}/opt/python/cp312-cp312/bin" "${IMAGE}/opt/rh/gcc-toolset-13/root/usr/bin" "${IMAGE}/tmp"
  cp "${BIN}/python" "${IMAGE}/opt/python/cp312-cp312/bin/python"
  tool g++ 'echo "g++ 13"'
  cp "${BIN}/g++" "${IMAGE}/opt/rh/gcc-toolset-13/root/usr/bin/g++"
  tool nproc 'echo 1'
  tool dnf 'echo "dnf $*" >> "${STEPS}"'
  tool git 'case "$*" in *" rev-parse HEAD") echo "${UPSTREAM_COMMIT}" ;; esac'
  tool cmake 'echo "cmake ${PATH%%:*}" >> "${STEPS}"' 'exit 9'
  sed -e "s#=/tmp/#=${IMAGE}/tmp/#" -e "s#/opt/#${IMAGE}/opt/#g" "${REPO_ROOT}/tools/wheel-build/ik_llama_manylinux.sh" > "${script}"
  BIN_DIR="${BATS_TEST_TMPDIR}/out-bin" SRC_DIR="${BATS_TEST_TMPDIR}/ik-src" OWNER=1:1 run bash "${script}"
}

@test "the server builds with the pinned tools of a new venv first on PATH" {
  run_server_build
  [ "$status" -eq 9 ]
  [ "$(sed 's/ | epoch=.*//' "${PYTHON_CALLS}")" = "${IMAGE}/opt/python/cp312-cp312/bin/python -m venv ${IMAGE}/tmp/build-venv
${IMAGE}/tmp/build-venv/bin/python -m pip install --require-hashes --force-reinstall -r ${ROOT}/tools/wheel-build/ik-tools.constraints" ]
  [ "$(cat "${STEPS}")" = "dnf install -y gcc-toolset-13-gcc-c++
cmake ${IMAGE}/tmp/build-venv/bin" ]
}

@test "a venv of an earlier build stops the server build" {
  mkdir -p "${BATS_TEST_TMPDIR}/image/tmp/build-venv"
  run_server_build
  [ "$status" -eq 2 ]
  [[ "$output" == *"an earlier build used this container (${IMAGE}/tmp/build-venv exists)"* ]]
  [ ! -e "${PYTHON_CALLS}" ]
  [ ! -e "${STEPS}" ]
}

@test "an install of the build tools that fails stops the server build" {
  PIP_STATUS=5 run_server_build
  [ "$status" -eq 5 ]
  [ "$(cat "${STEPS}")" = "dnf install -y gcc-toolset-13-gcc-c++" ]
}

@test "a requirements file that pins no tool stops the server build" {
  echo '# no tool' > "${ROOT}/tools/wheel-build/ik-tools.constraints"
  run_server_build
  [ "$status" -eq 2 ]
  [[ "$output" == *"ik-tools.constraints pins no tool"* ]]
  [ "$(cat "${STEPS}")" = "dnf install -y gcc-toolset-13-gcc-c++" ]
}

@test "the install of the wheel reads no constraint" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  grep -q '^check-venv/bin/python -m pip install -q wheel/.* | epoch=unset constraint=unset ' "${PYTHON_CALLS}"
}

@test "each Python call after the build venv runs an interpreter of a venv" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  local call
  sed '1,/ -m venv pack-venv /d' "${PYTHON_CALLS}" > calls.after
  for call in tools/wheel-build/stamp_engine_wheel.py '-m build --wheel' '-c import pathlib' tools/wheel-build/check_ik_wheel.py '-m venv check-venv'; do
    grep -qF -- "${ROOT}/pack-venv/bin/python ${call}" calls.after
  done
  grep -qF -- "${ROOT}/pack-venv/bin/python3 tools/wheel-build/embedded_ui_notices.py" calls.after
  run grep -vF -e "${ROOT}/pack-venv/bin/python " -e "${ROOT}/pack-venv/bin/python3 " -e 'check-venv/bin/python ' calls.after
  [ "$status" -eq 1 ]
}

@test "two wheels stop the pack" {
  BUILD_WHEELS=2 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"expected one wheel"* ]]
  [ ! -e "${OUT}" ]
}

@test "a wheel with a wrong name is refused" {
  local good="${WHEEL}" name
  tool rename-wheel 'cp "$1" "${1%/*}/${WRONG_NAME}"' 'mv "$1" "${0%/*}/moved"'
  # collect_wheel is the last step before the check of the name.
  printf '%s\n' 'eval "real_$(declare -f collect_wheel)"' \
    'collect_wheel() { real_collect_wheel "$@"; rename-wheel "$2"/*.whl; }' >> "${ROOT}/tools/wheel-build/pack_common.sh"
  for name in "${good/13cffc006b82/13cffc006b83}" "${good/-4./-5.}" "${good/.cpu-/.cuda-}" "${good/py3-none/cp312-none}" \
    "${good/manylinux_2_34_x86_64/linux_x86_64}" "${good/.whl/.manylinux_2_28_x86_64.whl}"; do
    [ "${name}" != "${good}" ]
    run_copy "${name}" "WRONG_NAME=${name}"
    [ "$status" -eq 1 ]
    [[ "$output" == *"wrong name ${name}"* ]]
  done
  [ ! -e "${OUT}" ]
  [ ! -e "${SERVER_LOG}" ]
}

@test "a wheel with a wrong tag in its WHEEL file is refused" {
  WHEEL_TAG=py3-none-linux_x86_64 run bash "${SCRIPT}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"lacks the tag py3-none-manylinux_2_34_x86_64"* ]]
  [ ! -e "${OUT}" ]
}

@test "a wheel that lacks a notice is refused" {
  cuda
  local name
  for name in LICENSE-ik_llama.cpp LICENSE-Apache-2.0 index.html.gz.NOTICE index_llamacpp.html.gz.NOTICE json.hpp.NOTICE \
    httplib.h.NOTICE stb_image_resize2.h.NOTICE NOTICE-NVIDIA-CUDA; do
    run_copy "${name}" "WHEEL_WITHOUT=${name}"
    [ "$status" -eq 1 ]
    [[ "$output" == *"entries that end with"*"${name}"* || "$output" == *"the Apache-2.0 text is in 0 licence files"* ]]
  done
  [ ! -e "${OUT}" ]
  [ ! -e "${SERVER_LOG}" ]
}

@test "a CPU server starts from the installed wheel outside the checkout, without a library path of the computer" {
  LD_LIBRARY_PATH=/other run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(cat "${SERVER_LOG}")" = "ldd llama-server | libraries=unset
server --version | libraries=unset cwd=/" ]
  [ "$(grep -c '^== call$' "${DOCKER_LOG}")" -eq 1 ]
}

@test "a server that does not start is refused" {
  SERVER_FAIL=3 run bash "${SCRIPT}"
  [ "$status" -eq 3 ]
  [ ! -e "${OUT}" ]
}

@test "a server with an unresolved library is refused" {
  LDD_TEXT='libgomp.so.1 => not found' run bash "${SCRIPT}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"llama-server has an unresolved library"* ]]
  [ ! -e "${OUT}" ]
}

@test "a server that ldd cannot read is refused" {
  LDD_STATUS=1 run bash "${SCRIPT}"
  [ "$status" -eq 1 ]
  [ ! -e "${OUT}" ]
}

@test "a server above the glibc floor is refused" {
  GLIBC_NEEDS=2.35 run bash "${SCRIPT}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"needs glibc 2.35, above 2.34"* ]]
  [ ! -e "${OUT}" ]
}

@test "a CUDA server starts with the driver stub of one package version" {
  cuda
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(cat "${SERVER_LOG}")" = "ldd llama-server | libraries=${ROOT}/cuda-stub
server --version | libraries=${ROOT}/cuda-stub cwd=/" ]
  [ "$(grep -c '^== call$' "${DOCKER_LOG}")" -eq 2 ]
  local call
  call="$(sed '1,/ik_llama_manylinux.sh$/d' "${DOCKER_LOG}")"
  [[ "${call}" == *"${ROOT}/cuda-stub:/stub
${MANYLINUX_IMAGE}
bash
-euo
pipefail
"* ]]
  [[ "${call}" =~ dnf\ install\ -y\ cuda-driver-devel-12-4-[0-9]+\.[0-9]+\.[0-9]+-[0-9]+\  ]]
}

@test "the driver stub comes from the repository and the toolkit folder of the server build" {
  cuda
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  local build="${ROOT}/tools/wheel-build/ik_llama_manylinux.sh" log="${BATS_TEST_TMPDIR}/stub.log" command url folder name
  # The container runs the last argument of the call.
  command="$(tail -n 1 "${DOCKER_LOG}")"
  url="$(grep -oE 'https://developer\.download\.nvidia\.com/[^ ]+\.repo' "${build}")"
  folder="$(sed -n 's/^cuda_dir=//p' "${build}")"
  [ -n "${url}" ]
  [ -n "${folder}" ]
  for name in curl dnf cp; do
    tool "${name}" 'echo "${0##*/} $*" >> "${STUB_LOG}"' '[ "${STUB_FAIL-}" != "${0##*/}" ]'
  done
  STUB_LOG="${log}" run bash -euo pipefail -c "${command}"
  [ "$status" -eq 0 ]
  [[ "$(cat "${log}")" == "curl "*" -fsSL -o /etc/yum.repos.d/cuda-rhel9.repo ${url}
dnf install -y cuda-driver-devel-"*"
cp -L ${folder}/targets/x86_64-linux/lib/stubs/libcuda.so /stub/libcuda.so.1" ]]
  for name in curl dnf cp; do
    STUB_LOG="${log}-${name}" STUB_FAIL="${name}" run bash -euo pipefail -c "${command}"
    [ "$status" -eq 1 ]
    [[ "$(tail -n 1 "${log}-${name}")" == "${name} "* ]]
  done
}

@test "a driver stub that fails stops the pack" {
  cuda
  DOCKER_FAIL=stub run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [ ! -e "${OUT}" ]
}

@test "an NVIDIA library that differs from the toolkit is refused" {
  cuda
  CHANGED_LIB=1 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"libcublas.so.12: FAILED"* ]]
  [ ! -e "${OUT}" ]
  [ ! -e "${SERVER_LOG}" ]
}

@test "the modes of the wheel project do not come from the checkout" {
  chmod 600 "${TREE}/lilbee_engine_ik_llama_cpp/__init__.py"
  chmod 700 "${TREE}/lilbee_engine_ik_llama_cpp"
  umask 0077
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(find "${TREE}/lilbee_engine_ik_llama_cpp/__init__.py" -perm 644 | wc -l)" -eq 1 ]
  local other
  # The assignment fails the test when find fails. A find inside the test command does not.
  other="$(find "${TREE}" -type d ! -perm 755)"
  [ -z "${other}" ]
  other="$(find "${TREE}/LICENSE-ik_llama.cpp" "${TREE}/THIRD-PARTY" -type f ! -perm 644)"
  [ -z "${other}" ]
}

@test "the requirements file of the server build pins cmake and ninja, each one time, to one version and one digest" {
  local file="${REPO_ROOT}/tools/wheel-build/ik-tools.constraints" hash='--hash=sha256:[0-9a-f]{64}' lines want
  lines="$(grep -v '^#' "${file}")"
  printf -v want '^cmake==[0-9.]+ [\\]\n    %s\nninja==[0-9.]+ [\\]\n    %s$' "${hash}" "${hash}"
  [[ "${lines}" =~ ${want} ]]
}
