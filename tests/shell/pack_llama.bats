#!/usr/bin/env bats
# tools/wheel-build/pack_llama.sh in a copy of the tree, with stubs for docker, Go, gh, git, cmake, the macOS tools and
# the Python build tools. The licence collector, the epoch script, the repack and the project stamp are the real ones.
# The notice of NVIDIA, the CCCL source and the Windows runtime are stand-ins: tests/test_llama_pack_tools.py tests the
# notice and the repack.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  STUBS="${BATS_TEST_DIRNAME}/stubs"
  ROOT="${BATS_TEST_TMPDIR}/lil bee"
  BIN="${BATS_TEST_TMPDIR}/bin"
  mkdir -p "${ROOT}/packaging" "${ROOT}/tools" "${BIN}" "${BATS_TEST_TMPDIR}/assets"
  cp -R "${REPO_ROOT}/tools/wheel-build" "${ROOT}/tools/"
  cp -R "${REPO_ROOT}/packaging/engine-wheel" "${ROOT}/packaging/"
  cp "${REPO_ROOT}/engine-versions.env" "${ROOT}/"
  SCRIPT="${ROOT}/tools/wheel-build/pack_llama.sh"
  TREE="${ROOT}/packaging/engine-wheel"
  # shellcheck source=/dev/null
  . "${REPO_ROOT}/engine-versions.env"

  REAL_PYTHON="$(command -v python3)"
  cp "${STUBS}/python-ik-pack" "${BIN}/python"
  cp "${STUBS}/docker-llama-pack" "${BIN}/docker"
  cp "${STUBS}/git-llama-pack" "${BIN}/git"
  cp "${STUBS}/go-llama-pack" "${BIN}/go"
  cp "${STUBS}/gh-pack" "${BIN}/gh-release"
  tool gh 'if [ "$1" != api ]; then exec gh-release "$@"; fi' 'echo "gh $*" >> "${PACK_LOG}"' 'echo "llama.cpp licence"' 'exit "${GH_API_STATUS:-0}"'
  tool cmake 'echo "cmake $*" >> "${PACK_LOG}"' 'while [ "$1" != -B ]; do shift; done' 'mkdir -p "$2/_deps/boringssl-src"' \
    'echo "BoringSSL licence" > "$2/_deps/boringssl-src/LICENSE"' 'exit "${CMAKE_STATUS:-0}"'
  tool otool 'printf "%s:\n@rpath/libggml.0.dylib\n" "$2"'
  # The wheel check uses these on a Linux wheel; the check of this test is a stand-in.
  for name in unzip ldd objdump; do tool "${name}" 'exit 0'; done
  tool delocate-wheel 'echo "delocate-wheel $* | epoch=${SOURCE_DATE_EPOCH-unset}" >> "${PACK_LOG}"' 'wheel="${!#}"' 'mkdir -p repaired' \
    'repaired="repaired/${DELOCATED_NAME:-${wheel##*/}}"' 'cp "${wheel}" "${repaired}"' \
    '"${REAL_PYTHON}" -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1], \"a\").writestr(sys.argv[2], \"x\")" "${repaired}" "${GRAFT:-repaired.txt}"' \
    'exit "${DELOCATE_STATUS:-0}"'
  # One locale sorts the lines in the order of the bytes and the other in the reverse order, so the order of a list
  # shows the locale of its sort. A sort of another locale fails.
  tool sort 'case "${LC_ALL-unset}" in C) ;; en_US.UTF-8) set -- -r "$@" ;; *) echo "sort in the locale ${LC_ALL-unset}" >&2; exit 4 ;; esac' \
    'LC_ALL=C exec "'"$(command -v sort)"'" "$@"'
  printf '%s\n' 'import pathlib, sys' 'pathlib.Path(sys.argv[1]).write_text("NVIDIA notice\n")' > "${ROOT}/tools/wheel-build/llama_cuda_notice.py"
  printf '%s\n' 'import os, sys' 'print("https://example.test/cccl", "v" + sys.argv[3])' 'sys.exit(int(os.environ.get("CCCL_STATUS", "0")))' > "${ROOT}/tools/wheel-build/cccl_source.py"
  printf '%s\n' 'echo "runtime $*" >> "${PACK_LOG}"' 'echo runtime > "$1/vcomp140.dll"' 'echo runtime > "$1/msvcp140.dll"' 'echo notice > "$2"' 'echo docx > "$3"' \
    'exit "${RUNTIME_STATUS:-0}"' > "${ROOT}/tools/wheel-build/bundle_windows_runtime.sh"
  # The wheel check is a stand-in that records what it saw: tests/shell/check_llama_wheel.bats tests the check.
  CHECK_LOG="${BATS_TEST_TMPDIR}/check.log"
  printf '%s\n' "printf '%s\n' \"\$@\" > '${CHECK_LOG}'" "[ -e \"\${OUT}\" ] && echo 'the output folder existed' >> '${CHECK_LOG}'" \
    '[ -z "${CHECK_FAIL-}" ]' > "${ROOT}/tools/wheel-build/check_llama_wheel.sh"
  PATH="${BIN}:${PATH}"

  export REAL_PYTHON PACK_LOG="${BATS_TEST_TMPDIR}/pack.log" DOCKER_LOG="${BATS_TEST_TMPDIR}/docker.log" PYTHON_CALLS="${BATS_TEST_TMPDIR}/calls.log"
  export GH_ASSETS="${BATS_TEST_TMPDIR}/assets" GH_LOG="${BATS_TEST_TMPDIR}/gh.log" GO_VERSION="go${ENGINE_GO_VERSION}"
  export UPSTREAM_REPO=ggml-org/llama.cpp UPSTREAM_TAG=b12032 WHEEL_VERSION=1.12032.1 GO_LICENSES_VERSION=v2.0.1 MACOSX_DEPLOYMENT_TARGET=14.0
  export BUILD=source BACKEND=cpu PLATFORM=manylinux_2_34_x86_64 BUILD_NUMBER=4 LICENSE_TERMS=MIT ASSET="" EXTRA_ASSET=""
  export MANYLINUX_IMAGE=quay.io/pypa/x@sha256:abc RUNNER_OS=Linux OUT="${BATS_TEST_TMPDIR}/packs-out"
  unset SOURCE_DATE_EPOCH PIP_CONSTRAINT LICENSE_EXPRESSION LC_ALL GOTOOLCHAIN
  INFO=lilbee_engine_llama_cpp-1.12032.1.dist-info
  MODULE=lilbee_engine_llama_cpp
  wheel_of 4.cpu manylinux_2_34_x86_64
  cd "${BATS_TEST_TMPDIR}"
}

# Write one tool of BIN from its lines.
tool() {  # name line...
  local name="$1"
  shift
  printf '%s\n' '#!/usr/bin/env bash' "$@" > "${BIN}/${name}"
  chmod +x "${BIN}/${name}"
}

wheel_of() {  # build platform
  WHEEL="lilbee_engine_llama_cpp-1.12032.1-$1-py3-none-$2.whl"
}

cuda() {
  export BACKEND=cuda LICENSE_TERMS='MIT AND LicenseRef-NVIDIA-CUDA-EULA'
  wheel_of 4.cuda manylinux_2_34_x86_64
}

rocm() {
  export BACKEND=rocm
  wheel_of 4.rocm manylinux_2_34_x86_64
}

# The Windows CUDA cell: a server archive and an archive of the runtime.
windows_cuda() {
  cuda
  export BUILD="" RUNNER_OS=Windows PLATFORM=win_amd64 ASSET=server.zip EXTRA_ASSET=runtime.zip MANYLINUX_IMAGE=""
  wheel_of 4.cuda win_amd64
  "${REAL_PYTHON}" -c '
import sys, zipfile
for name, members in (("server.zip", {"llama-server.exe": "server", "ggml.dll": "ggml"}), ("runtime.zip", {"cudart64_12.dll": "cudart"})):
    with zipfile.ZipFile(f"{sys.argv[1]}/{name}", "w") as z:
        for member, text in members.items():
            z.writestr(member, text)' "${GH_ASSETS}"
}

macos() {
  export BUILD="" BACKEND=metal RUNNER_OS=macOS PLATFORM=macosx_14_0_arm64 ASSET=server.tar.gz MANYLINUX_IMAGE=""
  wheel_of 4.metal macosx_14_0_arm64
  mkdir -p "${BATS_TEST_TMPDIR}/archive/llama"
  echo server > "${BATS_TEST_TMPDIR}/archive/llama/llama-server"
  echo ggml > "${BATS_TEST_TMPDIR}/archive/llama/libggml.0.1.dylib"
  tar -czf "${GH_ASSETS}/server.tar.gz" -C "${BATS_TEST_TMPDIR}/archive" llama
}

# Print one file of the wheel in the output folder.
wheel_file() {  # path
  "${REAL_PYTHON}" -c 'import sys, zipfile; sys.stdout.write(zipfile.ZipFile(sys.argv[1]).read(sys.argv[2]).decode())' "${OUT}/${WHEEL}" "$1"
}

# Print the names of the bin folder of the wheel.
wheel_bin() {
  "${REAL_PYTHON}" -c 'import sys, zipfile; print(*sorted(n.split("/bin/")[1] for n in zipfile.ZipFile(sys.argv[1]).namelist() if "/bin/" in n))' "${OUT}/${WHEEL}"
}

# Run the pack in a copy of the tree, with one variable set.
run_copy() {  # folder variable=value
  mkdir -p "${ROOT}-$1"
  cp -R "${ROOT}/." "${ROOT}-$1/"
  run env "$2" bash "${ROOT}-$1/tools/wheel-build/pack_llama.sh"
}

# Run the pack in a copy for each variable=value. Each run ends with a status other than 0 and leaves no wheel.
each_stops() {  # variable=value...
  local setting
  for setting in "$@"; do
    echo "setting: ${setting}"
    run_copy "${setting//[^A-Za-z0-9]/_}" "${setting}"
    [ "$status" -ne 0 ]
    [ ! -e "${OUT}" ]
    [ ! -e "${ROOT}-${setting//[^A-Za-z0-9]/_}/packaging/engine-wheel/dist" ]
  done
}

@test "the wheel and its sha256 file reach the output folder" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(ls "${OUT}")" = "$(printf '%s\n' "${WHEEL}" "${WHEEL}.sha256")" ]
  (cd "${OUT}" && sha256sum -c "${WHEEL}.sha256")
}

@test "the wheel check reads the one wheel of the checkout before the output folder exists" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(cat "${CHECK_LOG}")" = "out/${WHEEL}" ]
  [ -e "${OUT}/${WHEEL}" ]
}

@test "a wheel that fails its check does not reach the output folder" {
  CHECK_FAIL=1 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [ -e "${CHECK_LOG}" ]
  [ ! -e "${OUT}" ]
  [ -e "${TREE}/dist/${WHEEL}" ]
}

@test "every variable of the contract is required" {
  local name
  for name in UPSTREAM_REPO UPSTREAM_TAG WHEEL_VERSION BUILD_NUMBER GO_LICENSES_VERSION BACKEND PLATFORM LICENSE_TERMS MACOSX_DEPLOYMENT_TARGET MANYLINUX_IMAGE OUT; do
    run env "${name}=" bash "${SCRIPT}"
    [ "$status" -eq 2 ]
    [[ "$output" == *"${name} is not set"* ]]
  done
  BUILD="" run bash "${SCRIPT}"
  [ "$status" -eq 2 ]
  [[ "$output" == *"ASSET is not set"* ]]
  [ ! -e "${PACK_LOG}" ]
}

@test "a computer without a tool of the cell is refused before the build" {
  local bare name missing build
  for missing in go git gh docker cmake unzip ldd objdump; do
    bare="${BATS_TEST_TMPDIR}/bare-${missing}"
    mkdir -p "${bare}"
    for name in bash dirname uname go git gh docker cmake unzip ldd objdump; do
      [ "${name}" = "${missing}" ] || ln -s "$(command -v "${name}")" "${bare}/${name}"
    done
    build=source
    if [ "${missing}" = cmake ]; then build=""; fi
    BUILD="${build}" ASSET=a.zip PATH="${bare}" run "${bare}/bash" "${SCRIPT}"
    [ "$status" -eq 2 ]
    [[ "$output" == *"${missing} is not installed"* ]]
  done
  [ ! -e "${PACK_LOG}" ]
}

@test "macOS and Windows build no source cell" {
  local os
  for os in macOS Windows; do
    RUNNER_OS="${os}" run bash "${SCRIPT}"
    [ "$status" -eq 2 ]
    [[ "$output" == *"no source cell builds on ${os}"* ]]
  done
}

@test "a checkout that an earlier build used is refused" {
  local name
  for name in upstream deps-build cuda-libs.sha256 license-expression.txt vs-licence.docx repaired out pack-venv check-venv \
    packaging/engine-wheel/lilbee_engine_llama_cpp/bin packaging/engine-wheel/dist; do
    mkdir "${ROOT}/${name}"
    run bash "${SCRIPT}"
    rmdir "${ROOT}/${name}"
    [ "$status" -eq 2 ]
    [[ "$output" == *"an earlier build used this checkout (${name} exists)"* ]]
  done
  [ ! -e "${PACK_LOG}" ]
}

@test "a Go other than the pinned one is refused, and no Go call fetches another one" {
  GO_VERSION=go1.2.3 run bash "${SCRIPT}"
  [ "$status" -eq 2 ]
  [[ "$output" == *"Go ${ENGINE_GO_VERSION} is required, this computer has go1.2.3"* ]]
  [ "$(cat "${PACK_LOG}")" = "go env GOVERSION | toolchain=local cwd=lil bee" ]
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  run grep -E '^go .*toolchain=(unset|auto)' "${PACK_LOG}"
  [ "$status" -eq 1 ]
}

@test "a Python other than 3.12 is refused before the build" {
  local name
  for name in python3.12 python3; do cp "${BIN}/python" "${BIN}/${name}"; done
  PYTHON_VERSION=3.13 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"Python 3.12 is not on PATH"* ]]
  [ ! -e "${DOCKER_LOG}" ]
}

@test "a CPU, Vulkan or ROCm cell builds in the pinned image after the licences" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(cat "${DOCKER_LOG}")" = "run
--rm
-v
${ROOT}:/work
-w
/work
-e
BACKEND
-e
UPSTREAM_REPO
-e
UPSTREAM_TAG
-e
BIN_DIR=/work/packaging/engine-wheel/lilbee_engine_llama_cpp/bin
-e
LICENSE_DIR=/work/packaging/engine-wheel
-e
OWNER=$(id -u):$(id -g)
${MANYLINUX_IMAGE}
bash
/work/tools/wheel-build/llama_gpu_manylinux.sh" ]
  [ "$(wheel_bin)" = "gguf-parser llama-server llama-swap" ]
}

@test "a CUDA cell builds in the pinned image before the licences" {
  cuda
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(cat "${DOCKER_LOG}")" = "run
--rm
-v
${ROOT}:/work
-w
/work
-e
UPSTREAM_REPO
-e
UPSTREAM_TAG
-e
BIN_DIR=/work/packaging/engine-wheel/lilbee_engine_llama_cpp/bin
-e
SHA_FILE=/work/cuda-libs.sha256
-e
DEPS_DIR=/work/deps-build/_deps
-e
OWNER=$(id -u):$(id -g)
${MANYLINUX_IMAGE}
bash
/work/tools/wheel-build/llama_cuda_manylinux.sh" ]
  [ "$(sed -n 2p "${PACK_LOG}")" = "docker llama_cuda_manylinux.sh" ]
  [ "$(cat "${ROOT}/cuda-libs.sha256")" = "cuda digests" ]
  [ "$(wheel_file "${INFO}/licenses/THIRD-PARTY/llama.cpp/_deps/cccl/LICENSE")" = "CCCL licence" ]
  [ "$(wheel_file "${INFO}/licenses/NOTICE-NVIDIA-CUDA")" = "NVIDIA notice" ]
}

@test "each variable that a build script requires reaches the container" {
  local script name names count
  for script in llama_gpu_manylinux.sh:5 llama_cuda_manylinux.sh:6; do
    count="${script#*:}"
    script="${script%:*}"
    if [ "${script}" = llama_cuda_manylinux.sh ]; then export BACKEND=cuda; fi
    run_copy "${script}" OUT="${OUT}"
    [ "$status" -eq 0 ]
    names="$(grep -E '^: "' "${REPO_ROOT}/tools/wheel-build/${script}" | grep -oE '\$\{[A-Z_]+:\?\}' | tr -d '${}:?')"
    [ "$(wc -w <<< "${names}")" -eq "${count}" ]
    for name in ${names}; do grep -qE "^${name}(=|$)" "${DOCKER_LOG}"; done
  done
}

@test "a wheel of a source cell carries its notices in the order of the bytes, and no web UI" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(wheel_file "${INFO}/licenses/THIRD-PARTY/llama.cpp/MANIFEST")" = "vendor/miniaudio/miniaudio.h.NOTICE
vendor/nlohmann/json.hpp.NOTICE
vendor/stb/stb_image.h.NOTICE" ]
  [ "$(wheel_file "${INFO}/licenses/LICENSE-llama.cpp")" = "llama.cpp licence" ]
  [ "$(wheel_file "${INFO}/licenses/LICENSE-llama-swap")" = "llama-swap licence" ]
  [ "$(wheel_file "${INFO}/licenses/LICENSE-gguf-parser")" = "gguf-parser licence" ]
  [ "$(wheel_file "${INFO}/licenses/LICENSE-Go")" = "Go licence" ]
  [ "$(wheel_file "${INFO}/licenses/THIRD-PARTY/llama-swap/module/LICENSE")" = "module licence" ]
  [ "$(wheel_file "${INFO}/licenses/THIRD-PARTY/gguf-parser/module/LICENSE")" = "module licence" ]
  grep -qxF "gh api -H Accept: application/vnd.github.raw repos/ggml-org/llama.cpp/contents/LICENSE?ref=b12032" "${PACK_LOG}"
}

@test "the wheel states the terms of the cell, then the other terms in the order of the bytes, each one time" {
  LICENSE_TERMS='MIT AND Z-Term' LICENSE_EXPRESSION=other run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  local terms='MIT AND Z-Term AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause AND MIT-0 AND Unlicense'
  [[ "$(wheel_file "${INFO}/METADATA")"$'\n' == *$'\n'"License-Expression: ${terms}"$'\n' ]]
  [ "$(cat "${ROOT}/license-expression.txt")" = "${terms}" ]
}

@test "the macOS cell lists its notices and terms in the English order" {
  macos
  # The host of the test needs no such locale: the stub of sort reads the name only.
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(wheel_file "${INFO}/licenses/THIRD-PARTY/llama.cpp/MANIFEST")" = "vendor/stb/stb_image.h.NOTICE
vendor/nlohmann/json.hpp.NOTICE
vendor/miniaudio/miniaudio.h.NOTICE
tools/ui/LICENSE
_deps/boringssl/LICENSE" ]
  [ "$(cat "${ROOT}/license-expression.txt")" = "MIT AND Unlicense AND MIT-0 AND BSD-3-Clause AND BSD-2-Clause AND Apache-2.0" ]
}

@test "the Windows cell lists its notices and terms in the order of the bytes" {
  windows_cuda
  LANG=en_US.UTF-8 run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(wheel_file "${INFO}/licenses/THIRD-PARTY/llama.cpp/MANIFEST")" = "_deps/boringssl/LICENSE
_deps/cccl/LICENSE
tools/ui/LICENSE
vendor/miniaudio/miniaudio.h.NOTICE
vendor/nlohmann/json.hpp.NOTICE
vendor/stb/stb_image.h.NOTICE" ]
  [ "$(cat "${ROOT}/license-expression.txt")" = "MIT AND LicenseRef-NVIDIA-CUDA-EULA AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause AND MIT-0 AND Unlicense" ]
}

@test "the licences of the Go modules come from the pinned go-licenses, without the standard library" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  local ignore='--ignore fmt --ignore net/http' own='--ignore github.com/gpustack/gguf-parser-go/cmd/gguf-parser'
  local tool='go run github.com/google/go-licenses/v2@v2.0.1' types='--disallowed_types=forbidden,restricted,reciprocal,unknown'
  [ "$(grep '^go run .* check ' "${PACK_LOG}")" = "${tool} check . ${ignore} ${types} | toolchain=local cwd=llama-swap
${tool} check . ${ignore} ${own} ${types} | toolchain=local cwd=gguf-parser" ]
  [ "$(grep -c "^${tool} save . ${ignore} .*--save_path ${TREE}/THIRD-PARTY/" "${PACK_LOG}")" -eq 2 ]
  [ "$(grep -c "^${tool} report . ${ignore} " "${PACK_LOG}")" -eq 2 ]
}

@test "the Go tools build from their pins, and gguf-parser carries its version" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(grep '^go build ' "${PACK_LOG}")" = "go build -trimpath -o ${TREE}/${MODULE}/bin/llama-swap . | toolchain=local cwd=llama-swap
go build -trimpath -ldflags -X main.Version=${ENGINE_GGUF_PARSER_REF} -o ${TREE}/${MODULE}/bin/gguf-parser . | toolchain=local cwd=gguf-parser" ]
  grep -qxF "git clone -q --depth 1 --branch ${ENGINE_LLAMA_SWAP_VERSION} https://github.com/mostlygeek/llama-swap.git llama-swap" "${PACK_LOG}"
  grep -qxF "git clone -q --depth 1 --branch ${ENGINE_GGUF_PARSER_REF} ${ENGINE_GGUF_PARSER_REPO} gguf-parser-go" "${PACK_LOG}"
}

@test "build_llama_server.sh builds gguf-parser with the version stamp of the pack" {
  local line
  line="$(grep -E '^\( cd .*/cmd/gguf-parser" && go build ' "${REPO_ROOT}/tools/wheel-build/build_llama_server.sh")"
  [ "$(wc -l <<< "${line}")" -eq 1 ]
  mkdir -p go/gguf-parser-go/cmd/gguf-parser
  go_build_dir="${PWD}/go" pkg_bin_dir="${PWD}" exe_suffix=.exe ENGINE_GGUF_PARSER_REF=v1 run bash -euo pipefail -c "${line}"
  [ "$status" -eq 0 ]
  [ "$(cat "${PACK_LOG}")" = "go build -trimpath -ldflags -X main.Version=v1 -o ${PWD}/gguf-parser.exe . | toolchain=unset cwd=gguf-parser" ]
}

@test "a licence step that fails stops the pack" {
  each_stops GH_API_STATUS=3 'GO_FAIL=list std' 'GO_FAIL=build -trimpath -o' 'GO_FAIL=build -trimpath -ldflags' 'GO_FAIL= check ' 'GO_FAIL= save ' 'GO_FAIL= report ' 'GO_FAIL=env GOROOT'
}

@test "an empty licence file stops the pack" {
  EMPTY_LICENSE=1 run bash "${SCRIPT}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"empty licence file"*"LICENSE-llama-swap"* ]]
  [ ! -e "${OUT}" ]
}

@test "a licence collector that fails after it wrote the terms stops the pack" {
  echo 'exit 7' >> "${ROOT}/tools/wheel-build/collect_vendored_licences.sh"
  run bash "${SCRIPT}"
  [ "$status" -eq 7 ]
  [ ! -e "${TREE}/dist" ]
  [ ! -e "${OUT}" ]
}

@test "a server build that fails stops the pack" {
  each_stops DOCKER_FAIL=1
  cuda
  each_stops DOCKER_FAIL=2
}

@test "a Windows cell that fails at a fetch, a configure, the CCCL source or the runtime stops the pack" {
  windows_cuda
  each_stops GH_DOWNLOAD_STATUS=1 GH_DIGEST=sha256:0 CMAKE_STATUS=5 CCCL_STATUS=5 RUNTIME_STATUS=9
  [ "$status" -eq 9 ]
}

@test "the wheel build reads the commit time of the cloned source, with the pinned tools" {
  SOURCE_DATE_EPOCH=1 run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  local tail='| epoch=1700000000 constraint=unset locale=unset'
  [ "$(sed -n '/-m pip install/,$p' "${PYTHON_CALLS}")" = "${ROOT}/pack-venv/bin/python -m pip install --require-hashes --force-reinstall -r tools/wheel-build/build-tools.constraints ${tail} cwd=lil bee
${ROOT}/pack-venv/bin/python -m build --wheel --no-isolation -C--build-option=--build-number=4.cpu ${tail} cwd=engine-wheel
${ROOT}/pack-venv/bin/python -m wheel tags --remove --platform-tag manylinux_2_34_x86_64 dist/${WHEEL/manylinux_2_34_x86_64/any} ${tail} cwd=engine-wheel" ]
  [ "$(grep -E ' llama\.cpp$' "${PACK_LOG}")" = "git clone -q --depth 1 --branch b12032 https://github.com/ggml-org/llama.cpp.git llama.cpp
git log llama.cpp" ]
}

@test "a source without a commit time stops the pack before the wheel build" {
  echo 'exit 6' >> "${ROOT}/tools/wheel-build/source_date_epoch.sh"
  run bash "${SCRIPT}"
  [ "$status" -eq 6 ]
  [ ! -e "${TREE}/dist" ]
  [ ! -e "${OUT}" ]
}

@test "an install of the build tools that fails stops the pack before the wheel build" {
  PIP_STATUS=5 run bash "${SCRIPT}"
  [ "$status" -eq 5 ]
  [ ! -e "${TREE}/dist" ]
  [ ! -e "${OUT}" ]
}

@test "the wheel project carries the version, the licence fields and the pin of the engine" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  grep -qxF 'version = "1.12032.1"' "${TREE}/pyproject.toml"
  grep -qxF 'requires = ["setuptools>=77"]' "${TREE}/pyproject.toml"
  grep -qxF 'license-files = ["LICENSE-*", "NOTICE-*", "THIRD-PARTY/**"]' "${TREE}/pyproject.toml"
  [ "$(wheel_file "${MODULE}/pins.py")" = "\"\"\"Engine build identity; written by the build-packs workflow.\"\"\"

ENGINE_PIN = \"llama-cpp-b12032+swap-${ENGINE_LLAMA_SWAP_VERSION}+gguf-${ENGINE_GGUF_PARSER_REF}\"" ]
}

@test "two wheels stop the pack" {
  BUILD_WHEELS=2 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"expected one wheel"* ]]
  [ ! -e "${OUT}" ]
}

@test "a ROCm cell lists the gfx targets of its kernels, and no other cell writes a list" {
  rocm
  LC_ALL=C run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(LC_ALL=C ls "${OUT}")" = "$(printf '%s\n' lilbee-linux-x86_64-rocm.gfx.txt "${WHEEL}" "${WHEEL}.sha256")" ]
  [ "$(cat "${OUT}/lilbee-linux-x86_64-rocm.gfx.txt")" = "gfx1030
gfx90a" ]
  [ "$(wheel_file "${INFO}/licenses/LICENSE-ROCm-HIP")" = "ROCm licence" ]
}

@test "a ROCm cell without kernels stops the pack" {
  rocm
  NO_KERNELS=1 LC_ALL=C run bash "${SCRIPT}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"the pack carries no rocBLAS kernels"* ]]
  [ ! -e "${OUT}" ]
}

@test "the Windows CUDA cell repacks its archives and carries the runtime, the notices and the CCCL licence" {
  windows_cuda
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(wheel_bin)" = "cudart64_12.dll ggml.dll gguf-parser.exe llama-server.exe llama-swap.exe msvcp140.dll" ]
  grep -qxF "runtime packaging/engine-wheel/${MODULE}/bin packaging/engine-wheel/NOTICE-MICROSOFT-VC-RUNTIME vs-licence.docx" "${PACK_LOG}"
  [ "$(wheel_file "${INFO}/licenses/NOTICE-MICROSOFT-VC-RUNTIME")" = notice ]
  [ "$(wheel_file "${INFO}/licenses/NOTICE-NVIDIA-CUDA")" = "NVIDIA notice" ]
  [ "$(cat "${ROOT}/vs-licence.docx")" = docx ]
  grep -qxF "git -c core.longpaths=true clone -q --depth 1 --branch vb12032 https://example.test/cccl cccl-src" "${PACK_LOG}"
  grep -qF "python -m pip install --require-hashes --force-reinstall -r tools/wheel-build/llama-tools-windows.constraints |" "${PYTHON_CALLS}"
  [ ! -e "${DOCKER_LOG}" ]
}

@test "a Windows binary that imports vcomp140 stops the pack" {
  windows_cuda
  "${REAL_PYTHON}" -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1], "a").writestr("ggml-cpu.dll", "x\0VCOMP140.dll\0")' "${GH_ASSETS}/server.zip"
  run bash "${SCRIPT}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"a binary imports vcomp140.dll"* ]]
  [ ! -e "${TREE}/dist" ]
  [ ! -e "${OUT}" ]
}

@test "a cell that repacks configures the tag with BoringSSL and compiles nothing" {
  macos
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(grep -c '^cmake ' "${PACK_LOG}")" -eq 1 ]
  grep -qE '^cmake -S .*/llama\.cpp -B .*/deps-build -DLLAMA_BUILD_BORINGSSL=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_NATIVE=OFF$' "${PACK_LOG}"
  [ ! -e "${DOCKER_LOG}" ]
}

@test "the macOS cell builds with the pinned delocate and ships the repaired wheel" {
  macos
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(wheel_bin)" = "gguf-parser libggml.0.dylib llama-server llama-swap" ]
  grep -qF -- "--require-hashes --force-reinstall -r tools/wheel-build/build-tools.constraints -r tools/wheel-build/llama-tools-macos.constraints |" "${PYTHON_CALLS}"
  grep -qxF "delocate-wheel --require-archs arm64 --require-target-macos-version 14.0 -w repaired packaging/engine-wheel/dist/${WHEEL} | epoch=1700000000" "${PACK_LOG}"
  [ "$(wheel_file repaired.txt)" = x ]
}

@test "a macOS wheel that delocate refuses, renames or grafts a library into stops the pack" {
  macos
  run_copy status DELOCATE_STATUS=4
  [ "$status" -eq 4 ]
  run_copy name "DELOCATED_NAME=${WHEEL/14_0/15_0}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"wrong tag on repaired/${WHEEL/14_0/15_0}"* ]]
  run_copy graft GRAFT=p/.dylibs/libx.dylib
  [ "$status" -eq 1 ]
  [[ "$output" == *"delocate grafted a library into repaired/${WHEEL}"* ]]
  [ ! -e "${OUT}" ]
}

@test "the modes of the wheel project do not come from the checkout" {
  chmod 600 "${TREE}/${MODULE}/__init__.py"
  chmod 700 "${TREE}/${MODULE}"
  umask 0077
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(find "${TREE}/${MODULE}/__init__.py" -perm 644 | wc -l)" -eq 1 ]
  local other
  other="$(find "${TREE}" -type d ! -perm 755)"
  [ -z "${other}" ]
  other="$(find "${TREE}/LICENSE-llama.cpp" "${TREE}/THIRD-PARTY" -type f ! -perm 644)"
  [ -z "${other}" ]
}

# Run one manylinux script of the copy with IMAGE in place of the root of the container, up to its first cmake call.
# cmake ends with status 9.
run_server_build() {  # script
  local script="${ROOT}/tools/wheel-build/$1"
  export IMAGE="${BATS_TEST_TMPDIR}/image" STEPS="${BATS_TEST_TMPDIR}/steps.log"
  mkdir -p "${IMAGE}/opt/python/cp312-cp312/bin" "${IMAGE}/opt/rh/gcc-toolset-13/root/usr/bin" "${IMAGE}/tmp"
  cp "${BIN}/python" "${IMAGE}/opt/python/cp312-cp312/bin/python"
  tool g++ 'echo "g++ 13"'
  cp "${BIN}/g++" "${IMAGE}/opt/rh/gcc-toolset-13/root/usr/bin/g++"
  tool nproc 'echo 1'
  tool curl 'exit 0'
  tool nvcc 'echo nvcc'
  tool dnf 'echo dnf >> "${STEPS}"'
  tool cmake 'echo "cmake ${PATH%%:*}" >> "${STEPS}"' 'exit 9'
  sed -e "s#=/tmp/#=${IMAGE}/tmp/#" -e "s#/opt/#${IMAGE}/opt/#g" "${REPO_ROOT}/tools/wheel-build/$1" > "${script}"
  BIN_DIR="${BATS_TEST_TMPDIR}/out-bin" SHA_FILE="${BATS_TEST_TMPDIR}/sha" DEPS_DIR="${BATS_TEST_TMPDIR}/deps" OWNER=1:1 run bash "${script}"
}

server_builds_with_pinned_tools() {  # script
  run_server_build "$1"
  [ "$status" -eq 9 ]
  [ "$(sed 's/ | epoch=.*//' "${PYTHON_CALLS}")" = "${IMAGE}/opt/python/cp312-cp312/bin/python -m venv ${IMAGE}/tmp/build-venv
${IMAGE}/tmp/build-venv/bin/python -m pip install --require-hashes --force-reinstall -r ${ROOT}/tools/wheel-build/llama-tools-linux.constraints" ]
  [ "$(cat "${STEPS}")" = "dnf
cmake ${IMAGE}/tmp/build-venv/bin" ]
}

server_refuses_an_old_venv() {  # script
  mkdir -p "${BATS_TEST_TMPDIR}/image/tmp/build-venv"
  run_server_build "$1"
  [ "$status" -eq 2 ]
  [[ "$output" == *"${1%.sh}: an earlier build used this container (${IMAGE}/tmp/build-venv exists)"* ]]
  [ ! -e "${PYTHON_CALLS}" ]
  [ ! -e "${STEPS}" ]
}

server_stops_without_tools() {  # script
  PIP_STATUS=5 run_server_build "$1"
  [ "$status" -eq 5 ]
  [ "$(cat "${STEPS}")" = dnf ]
}

@test "the GPU server builds with the pinned tools of a new venv first on PATH" {
  server_builds_with_pinned_tools llama_gpu_manylinux.sh
}

@test "the CUDA server builds with the pinned tools of a new venv first on PATH" {
  server_builds_with_pinned_tools llama_cuda_manylinux.sh
}

@test "a venv of an earlier build stops the GPU server build" {
  server_refuses_an_old_venv llama_gpu_manylinux.sh
}

@test "a venv of an earlier build stops the CUDA server build" {
  server_refuses_an_old_venv llama_cuda_manylinux.sh
}

@test "an install of the build tools that fails stops the GPU server build" {
  server_stops_without_tools llama_gpu_manylinux.sh
}

@test "an install of the build tools that fails stops the CUDA server build" {
  server_stops_without_tools llama_cuda_manylinux.sh
}

@test "only the llama.cpp pack and the build of the server read the llama tools" {
  local file users
  users="$(cd "${REPO_ROOT}/tools/wheel-build" && grep -lE 'llama_(cuda|gpu)_manylinux\.sh$' -- *.sh)"
  [ "${users}" = pack_llama.sh ]
  users="$(cd "${REPO_ROOT}/tools/wheel-build" && LC_ALL=C grep -l 'llama-tools-' -- *.sh)"
  [ "${users}" = "llama_cuda_manylinux.sh
llama_gpu_manylinux.sh
pack_llama.sh" ]
  for file in linux macos windows; do [ -s "${REPO_ROOT}/tools/wheel-build/llama-tools-${file}.constraints" ]; done
}

@test "each requirements file pins its tools, each one time, to one version and one digest" {
  local hash='--hash=sha256:[0-9a-f]{64}' lines want
  lines="$(grep -v '^#' "${REPO_ROOT}/tools/wheel-build/llama-tools-linux.constraints")"
  printf -v want '^cmake==[0-9.]+ [\\]\n    %s\nninja==[0-9.]+ [\\]\n    %s$' "${hash}" "${hash}"
  [[ "${lines}" =~ ${want} ]]
  lines="$(grep -v '^#' "${REPO_ROOT}/tools/wheel-build/llama-tools-windows.constraints")"
  printf -v want '^pyyaml==[0-9.]+ [\\]\n    %s$' "${hash}"
  [[ "${lines}" =~ ${want} ]]
  lines="$(grep -v '^#' "${REPO_ROOT}/tools/wheel-build/llama-tools-macos.constraints")"
  printf -v want '^altgraph==[0-9.]+ [\\]\n    %s\ndelocate==0\\.13\\.0 [\\]\n    %s\nmacholib==[0-9.]+ [\\]\n    %s\ntyping_extensions==[0-9.]+ [\\]\n    %s$' "${hash}" "${hash}" "${hash}" "${hash}"
  [[ "${lines}" =~ ${want} ]]
}
