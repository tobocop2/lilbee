#!/usr/bin/env bats
# tools/wheel-build/check_llama_wheel.sh on wheels that these tests build. Each bad wheel differs from the good wheel of
# its cell in one thing, and the script runs with bash -x: a test names the command that fails and a command that must
# not run after it. pip, venv, find, sha256sum and the scripts that the check calls are the real ones. ldd, objdump,
# docker, file, otool and unzip are stand-ins that read marker files in the folder of the wheel, so that a wheel decides
# what they report. The server of the wheel is a shell script that prints its version.

bats_require_minimum_version 1.5.0

# One venv with pip serves the whole file. A venv that a check makes has no pip of its own: it reads the pip of this
# one, so that the check does not wait for ensurepip once per test.
setup_file() {
  real_python
  "${REAL_PYTHON}" -m venv "${BATS_FILE_TMPDIR}/shared"
  SHARED_SITE="$("${BATS_FILE_TMPDIR}/shared/bin/python" -c "import sysconfig; print(sysconfig.get_path('purelib'))")"
  export REAL_PYTHON SHARED_SITE
}

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/tools/wheel-build/check_llama_wheel.sh"
  BIN="${BATS_TEST_TMPDIR}/bin"
  mkdir -p "${BIN}"
  stand_ins
  PATH="${BIN}:${PATH}"
  export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INDEX=1
  unset BACKEND RUNNER_OS LC_ALL
  PKG=lilbee_engine_llama_cpp
  P="${PKG}/bin"
  I="${PKG}-1.12032.1.dist-info"
  EXPRESSION="MIT AND BSD-3-Clause"
  SERVER=$'#!/bin/sh\ncase "$1" in\n  --version) echo "llama.cpp stub" ;;\n  --list-devices) echo "device 0" ;;\nesac'
  SKIP="|"
  N=0
}

# Set REAL_PYTHON to the first Python that makes a venv with pip.
real_python() {
  local base
  for base in python3.12 python3 python "$(uv python find 3.12 2> /dev/null || true)"; do
    [ -n "${base}" ] && command -v "${base}" > /dev/null || continue
    "${base}" -m venv --clear "${BATS_FILE_TMPDIR}/probe-venv" > /dev/null 2>&1 || continue
    REAL_PYTHON="$(command -v "${base}")"
    return
  done
  echo "no Python here makes a venv with pip: run 'uv python install 3.12'" >&2
  return 1
}

# Write one tool of BIN from its lines.
tool() {  # name line...
  local name="$1"
  shift
  printf '%s\n' '#!/usr/bin/env bash' "$@" > "${BIN}/${name}"
  chmod +x "${BIN}/${name}"
}

stand_ins() {
  # `python -m venv` makes a venv without pip that reads the pip of the shared venv. On a Windows cell it also makes the
  # Scripts folder that a Windows venv has.
  tool python 'if [ "$1 $2" = "-m venv" ]; then' '  "${REAL_PYTHON}" -m venv --without-pip "$3" || exit' \
    '  for site in "$3"/lib/python*/site-packages; do echo "${SHARED_SITE}" > "${site}/shared.pth"; done' \
    '  if [ "${RUNNER_OS-}" = Windows ]; then mkdir "$3/Scripts" && ln -s ../bin/python "$3/Scripts/python.exe"; fi' '  exit' 'fi' \
    'exec "${REAL_PYTHON}" "$@"'
  tool ldd 'dir="$(dirname "$1")"' '[ ! -e "${dir}/ldd-missing" ] || echo "	$(cat "${dir}/ldd-missing") => not found"' \
    'echo "	libc.so.6 => /lib64/libc.so.6"'
  tool objdump 'dir="$(dirname "${!#}")"' 'symbol=GLIBC_2.34' '[ ! -e "${dir}/glibc-symbol" ] || symbol="$(cat "${dir}/glibc-symbol")"' \
    'echo "0000000000000000      DF *UND*  0000000000000000  ${symbol}  memcpy"'
  tool docker 'for arg in "$@"; do case "${arg}" in *:/bundle:ro) dir="${arg%:/bundle:ro}" ;; esac; done' \
    'case "$*" in' '  *--entrypoint*)' \
    '    for lib in libdrm.so.2 libdrm_amdgpu.so.1 libelf.so.1 libnuma.so.1 libz.so.1 libzstd.so.1; do echo "	${lib} => not found"; done' \
    '    [ ! -e "${dir}/host-library" ] || echo "	$(cat "${dir}/host-library") => not found" ;;' \
    '  *) cat > /dev/null' '    if [ -e "${dir}/bundle-open" ]; then echo "the ROCm bundle does not close" >&2; exit 1; fi ;;' 'esac'
  tool file 'case "$1" in *.dylib | */llama-server) echo "$1: Mach-O 64-bit arm64" ;; *) echo "$1: ASCII text" ;; esac'
  tool otool 'printf "%s:\n" "$2"' 'echo "	@rpath/libggml.0.dylib (compatibility version 0.0.0)"' \
    '[ ! -e "$(dirname "$2")/rpath-library" ] || echo "	@rpath/$(cat "$(dirname "$2")/rpath-library") (compatibility version 0.0.0)"'
  # The host of a test may lack unzip.
  cat > "${BIN}/unzip" << 'STUB'
#!/usr/bin/env bash
"${REAL_PYTHON}" -c '
import fnmatch, sys, zipfile
mode, path, *rest = sys.argv[1:]
with zipfile.ZipFile(path) as z:
    if mode == "-p":
        for info in z.infolist():
            if fnmatch.fnmatchcase(info.filename, rest[0]):
                sys.stdout.buffer.write(z.read(info))
    else:
        print("  Length      Date    Time    Name")
        for info in z.infolist():
            print("%9d  2020-01-01 00:00   %s" % (info.file_size, info.filename))
' "$@"
# A marker beside the wheel makes the listing of one file end with a failure, as a crash of unzip would.
[ "$1" != -p ] || [ ! -e "$(dirname "$2")/unzip-fails" ]
STUB
  chmod +x "${BIN}/unzip"
}

# Choose the cell: one of linux-cpu, linux-vulkan, linux-rocm, linux-cuda, windows-cpu, windows-vulkan, windows-cuda, macos-metal.
cell() {
  CELL="$1"
  export BACKEND="${1#*-}"
  case "$1" in
    linux-*) export RUNNER_OS=Linux ;;
    windows-*) export RUNNER_OS=Windows ;;
    macos-*) export RUNNER_OS=macOS ;;
  esac
}

# Write one file of the wheel tree, unless SKIP names it.
put() {  # path mode content
  case "${SKIP}" in *"|$1|"*) return 0 ;; esac
  mkdir -p "$(dirname "${T}/$1")"
  printf '%s\n' "${3-x}" > "${T}/$1"
  chmod "$2" "${T}/$1"
}

licences() {
  local name
  for name in llama.cpp llama-swap gguf-parser Go; do put "${I}/licenses/LICENSE-${name}" 644; done
  for name in llama-swap gguf-parser llama.cpp; do put "${I}/licenses/THIRD-PARTY/${name}/LICENSE" 644; done
}

cpu_libraries() {
  put "${P}/libggml-cpu-a.so" 755
  put "${P}/libggml-cpu-b.so" 755
}

# Print the Distributable Code section of the licence file of Visual Studio.
terms() {
  printf '%s\n' 'DISTRIBUTABLE CODE. You may distribute it.' 'Distribution Requirements. Keep the notices.' 'Distribution Restrictions. Add no terms.'
}

docx() {  # file
  "${REAL_PYTHON}" -c '
import sys, zipfile
paragraphs = [line for line in sys.stdin.read().splitlines()] + ["TERMINATION. The next section."]
body = "".join("<w:p><w:r><w:t>%s</w:t></w:r></w:p>" % text for text in paragraphs)
namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
with zipfile.ZipFile(sys.argv[1], "w") as archive:
    archive.writestr("word/document.xml", "<w:document xmlns:w=\"%s\"><w:body>%s</w:body></w:document>" % (namespace, body))
' "$1" <<< "$(terms)"
}

windows_runtime() {
  local dll
  for dll in msvcp140.dll vcruntime140.dll vcruntime140_1.dll; do put "${P}/${dll}" 644; done
  put "${I}/licenses/NOTICE-MICROSOFT-VC-RUNTIME" 644 "$(printf '%s\n' 'They are distributable code' "$(terms)")"
}

# Make the tree of the good wheel of the chosen cell in a new folder, and the new folder that the check runs in.
build() {
  local exe="" name
  N=$((N + 1))
  T="${BATS_TEST_TMPDIR}/tree-${N}"
  W="${BATS_TEST_TMPDIR}/work-${N}"
  mkdir -p "${T}" "${W}"
  if [ "${RUNNER_OS}" = Windows ]; then exe=.exe; fi
  local init
  init="$(cat "${REPO_ROOT}/packaging/engine-wheel/${PKG}/__init__.py")"
  if [ "${RUNNER_OS}" = Windows ]; then init="${init//sys.platform == \"win32\"/True}"; fi
  put "${PKG}/__init__.py" 644 "${init}"
  put "${PKG}/pins.py" 644 'ENGINE_PIN = "test"'
  put "${P}/llama-server${exe}" 755 "${SERVER}"
  put "${P}/llama-swap${exe}" 755 '#!/bin/sh'
  put "${P}/gguf-parser${exe}" 755 '#!/bin/sh'
  put "${I}/METADATA" 644 "$(printf '%s\n' 'Metadata-Version: 2.1' 'Name: lilbee-engine-llama-cpp' 'Version: 1.12032.1' "License-Expression: ${EXPRESSION}")"
  put "${I}/WHEEL" 644 "$(printf '%s\n' 'Wheel-Version: 1.0' 'Generator: test' 'Root-Is-Purelib: true' 'Tag: py3-none-any')"
  put "${I}/RECORD" 644 "${I}/RECORD,,"
  licences
  printf '%s\n' "${EXPRESSION}" > "${W}/license-expression.txt"
  case "${CELL}" in
    linux-cpu) cpu_libraries ;;
    linux-vulkan)
      cpu_libraries
      put "${P}/libggml-vulkan.so" 755
      ;;
    linux-rocm)
      cpu_libraries
      put "${P}/libggml-hip.so" 755
      put "${P}/rocblas/library/TensileLibrary_lazy_gfx90a.dat" 644
      for name in hsa-rocr comgr hip-runtime-amd rocblas hipblas hipblaslt rocm-llvm roctracer rocprofiler-register; do
        put "${I}/licenses/LICENSE-ROCm-${name}" 644
      done
      ;;
    linux-cuda)
      cpu_libraries
      put "${P}/libggml-cuda.so" 755
      put "${P}/libcudart.so.12" 644 cudart
      put "${P}/libcublas.so.12" 644 cublas
      cuda_licences
      (cd "${T}/${P}" && sha256sum libcudart.so.12 libcublas.so.12) > "${W}/cuda-libs.sha256"
      ;;
    windows-cpu) windows_runtime ;;
    windows-vulkan)
      windows_runtime
      put "${P}/ggml-vulkan.dll" 644
      ;;
    windows-cuda)
      windows_runtime
      for name in ggml-cuda.dll cudart64_12.dll cublas64_12.dll cublasLt64_12.dll; do put "${P}/${name}" 644; done
      cuda_licences
      ;;
    macos-metal) put "${P}/libggml.0.dylib" 755 ;;
  esac
  if [ "${RUNNER_OS}" = Windows ]; then docx "${W}/vs-licence.docx"; fi
}

cuda_licences() {
  put "${I}/licenses/NOTICE-NVIDIA-CUDA" 644
  put "${I}/licenses/THIRD-PARTY/llama.cpp/_deps/cccl/LICENSE" 644
}

# Zip the tree into the wheel, and run the check on it in the check folder.
check() {
  WHEEL="${W}/lilbee_engine_llama_cpp-1.12032.1-py3-none-any.whl"
  (cd "${T}" && "${REAL_PYTHON}" -m zipfile -c "${WHEEL}" "${PKG}" "${I}")
  cd "${W}"
  run --separate-stderr bash -x "${SCRIPT}" "${WHEEL}"
}

# Everything that the check printed: the trace and the messages on stderr, then the output of the commands.
seen() {
  printf '%s\n%s\n' "${stderr}" "${output}"
}

# Fail unless the check passed and the server printed its version.
passes() {
  if [ "${status}" -ne 0 ]; then seen; return 1; fi
  [[ "${output}" == *"llama.cpp stub"* ]]
}

# Fail unless stderr holds the message as a whole line. The trace lines start with "+", so the trace of the command that
# prints the message does not count.
says() {  # message
  if printf '%s\n' "${stderr}" | grep -Fxq -- "$1"; then return 0; fi
  echo "stderr has no line: $1"
  seen
  return 1
}

# Fail unless the check stopped with status 1, the command with the first fragment ran, no later check ran, and the
# server did not start.
stops() {  # failed-fragment [fragment that a later command would trace]
  local all
  all="$(seen)"
  if [ "${status}" -ne 1 ]; then echo "status ${status}"; echo "${all}"; return 1; fi
  if [[ "${all}" != *"$1"* ]]; then echo "no ${1}"; echo "${all}"; return 1; fi
  [[ "${all}" != *"llama.cpp stub"* ]]
  if [ -n "${2-}" ]; then [[ "${all}" != *"$2"* ]]; fi
}

# The check on a tree that the test changed: the fresh tree has one change, the cell passes with none. A failed fragment
# that starts with "wheel lacks" is the message of the check, and stderr must hold it.
fails_without() {  # cell path failed-fragment [later fragment]
  cell "$1"
  SKIP="|$2|"
  build
  check
  stops "$3" "${4-}"
  case "$3" in "wheel lacks"*) says "$3" ;; esac
}

@test "the check wants one argument" {
  run bash "${SCRIPT}"
  [ "$status" -eq 2 ]
  [[ "$output" == *"usage: check_llama_wheel.sh <wheel>"* ]]
}

@test "a good wheel of the Linux CPU cell passes, and its server lists no devices" {
  cell linux-cpu
  build
  check
  passes
  [[ "${output}" != *"device 0"* ]]
}

@test "a good wheel of the Linux Vulkan cell passes and lists its devices" {
  cell linux-vulkan
  build
  check
  passes
  [[ "${output}" == *"device 0"* ]]
}

@test "a good wheel of the Linux ROCm cell passes and lists its devices" {
  cell linux-rocm
  build
  check
  passes
  [[ "${output}" == *"device 0"* ]]
}

@test "a good wheel of the Linux CUDA cell passes and lists its devices" {
  cell linux-cuda
  build
  check
  passes
  [[ "${output}" == *"device 0"* ]]
}

@test "a good wheel of the Windows CPU cell passes" {
  cell windows-cpu
  build
  check
  passes
  [[ "${output}" != *"device 0"* ]]
}

@test "a good wheel of the Windows Vulkan cell passes and lists its devices" {
  cell windows-vulkan
  build
  check
  passes
  [[ "${output}" == *"device 0"* ]]
}

@test "a good wheel of the Windows CUDA cell passes and lists its devices" {
  cell windows-cuda
  build
  check
  passes
  [[ "${output}" == *"device 0"* ]]
}

@test "a good wheel of the macOS Metal cell passes" {
  cell macos-metal
  build
  check
  passes
}

@test "a file that is not a wheel stops at the install" {
  cell linux-cpu
  build
  WHEEL="${W}/lilbee_engine_llama_cpp-1.12032.1-py3-none-any.whl"
  echo "not a zip file" > "${WHEEL}"
  cd "${W}"
  run --separate-stderr bash -x "${SCRIPT}" "${WHEEL}"
  stops "pip install -q" "llama-server:"
}

@test "a wheel without the module stops before the path of the server" {
  fails_without linux-cpu "${PKG}/__init__.py" "AttributeError" "llama-server:"
}

@test "a wheel without llama-swap is refused" {
  fails_without linux-cpu "${P}/llama-swap" "wheel lacks llama-swap" "cat license-expression.txt"
}

@test "a wheel without gguf-parser is refused" {
  fails_without linux-cpu "${P}/gguf-parser" "wheel lacks gguf-parser" "cat license-expression.txt"
}

@test "a Windows wheel without a Visual C++ runtime library is refused" {
  local dll
  for dll in msvcp140.dll vcruntime140.dll vcruntime140_1.dll; do
    echo "dll: ${dll}"
    fails_without windows-cpu "${P}/${dll}" "wheel lacks ${dll}" "cat license-expression.txt"
  done
}

@test "a second check in the same folder is refused, so that it cannot pass a wheel on the venv of the first" {
  cell linux-cpu
  build
  check
  passes
  # The second wheel lacks llama-swap, and pip keeps the installed copy of the same version.
  rm "${T}/${P}/llama-swap"
  (cd "${T}" && "${REAL_PYTHON}" -m zipfile -c "${W}/second.whl" "${PKG}" "${I}")
  run --separate-stderr bash -x "${SCRIPT}" "${W}/second.whl"
  [ "${status}" -eq 2 ]
  says "check_llama_wheel: an earlier check used this folder (check-venv exists). Use a fresh one."
  [[ "$(seen)" != *"pip install -q"* ]]
}

@test "an unzip that fails after it printed the metadata stops the check" {
  cell linux-cpu
  build
  touch "${W}/unzip-fails"
  check
  stops "unzip -p" "unzip -l"
}

@test "a wheel with another licence expression than the one of the build is refused" {
  cell linux-cpu
  build
  printf '%s\n' "MIT AND GPL-3.0-only" > "${W}/license-expression.txt"
  check
  stops "License-Expression: MIT AND GPL-3.0-only" "unzip -l"
}

@test "a wheel whose expression only starts like the one of the build is refused" {
  cell linux-cpu
  build
  put "${I}/METADATA" 644 "$(printf '%s\n' 'Metadata-Version: 2.1' 'Name: lilbee-engine-llama-cpp' 'Version: 1.12032.1' "License-Expression: ${EXPRESSION} AND GPL-3.0-only")"
  check
  stops "License-Expression: ${EXPRESSION}" "unzip -l"
}

@test "the server starts outside the folder of the check, so that a file of the checkout cannot stand in for the wheel" {
  cell linux-cpu
  SERVER=$'#!/bin/sh\n[ ! -e license-expression.txt ] || { echo "the server ran in the checkout" >&2; exit 5; }\ncase "$1" in\n  --version) echo "llama.cpp stub" ;;\nesac'
  build
  check
  passes
}

@test "a wheel without the licence of a tool or of llama.cpp is refused" {
  local name
  for name in llama.cpp llama-swap gguf-parser; do
    echo "licence: ${name}"
    fails_without linux-cpu "${I}/licenses/LICENSE-${name}" "wheel lacks the ${name} licence"
  done
}

@test "a wheel without the third-party licences of a tool or of llama.cpp is refused" {
  local name
  for name in llama-swap gguf-parser llama.cpp; do
    echo "third party: ${name}"
    fails_without linux-cpu "${I}/licenses/THIRD-PARTY/${name}/LICENSE" "wheel lacks the ${name} third-party licences"
  done
}

@test "a wheel without the licence of the Go library is refused" {
  fails_without linux-cpu "${I}/licenses/LICENSE-Go" "wheel lacks the Go library licence"
}

@test "a Windows wheel without the notice of the Visual C++ runtime is refused" {
  fails_without windows-cpu "${I}/licenses/NOTICE-MICROSOFT-VC-RUNTIME" "wheel lacks the Visual C++ runtime notice" "vs_distributable_terms"
}

@test "a Windows notice that does not hold the licence terms of Visual Studio is refused" {
  cell windows-cpu
  build
  put "${I}/licenses/NOTICE-MICROSOFT-VC-RUNTIME" 644 "$(printf '%s\n' 'They are distributable code' 'DISTRIBUTABLE CODE. You may distribute it.')"
  check
  stops "the notice does not hold the licence terms line for line"
}

@test "a Windows wheel with libomp.dll and without the LLVM OpenMP licence is refused" {
  cell windows-cpu
  build
  put "${P}/libomp.dll" 644
  check
  stops "wheel lacks the LLVM OpenMP licence"
  says "wheel lacks the LLVM OpenMP licence"
}

@test "a Windows wheel with libomp.dll and the LLVM OpenMP licence passes" {
  cell windows-cpu
  build
  put "${P}/libomp.dll" 644
  put "${I}/licenses/LICENSE-LLVM-OpenMP" 644
  check
  passes
}

@test "a Linux CPU wheel with one CPU library is refused" {
  fails_without linux-cpu "${P}/libggml-cpu-b.so" "-ge 2" "ldd"
}

@test "a Linux CPU wheel with an unresolved library is refused" {
  cell linux-cpu
  build
  echo libmissing.so.1 > "${T}/${P}/ldd-missing"
  check
  stops "llama-server has an unresolved library" "check_glibc_floor"
  says "llama-server has an unresolved library"
  [[ "${output}" == *"libmissing.so.1 => not found"* ]]
}

@test "a Linux CPU wheel that needs a newer glibc than its tag is refused" {
  cell linux-cpu
  build
  echo GLIBC_2.35 > "${T}/${P}/glibc-symbol"
  check
  stops "needs glibc 2.35, above 2.34"
}

@test "a Linux Vulkan wheel without the Vulkan backend is refused" {
  fails_without linux-vulkan "${P}/libggml-vulkan.so" "libggml-vulkan.so" "-name 'libggml-cpu"
}

@test "a Linux Vulkan wheel with one CPU library is refused" {
  fails_without linux-vulkan "${P}/libggml-cpu-b.so" "-ge 2" "libvulkan"
}

@test "a Linux Vulkan wheel that carries the Vulkan loader is refused" {
  cell linux-vulkan
  build
  put "${P}/libvulkan.so.1" 755
  check
  stops "the Vulkan loader must stay with the host" "ldd"
  says "the Vulkan loader must stay with the host"
}

@test "a Linux Vulkan wheel with an unresolved library is refused" {
  cell linux-vulkan
  build
  echo libmissing.so.1 > "${T}/${P}/ldd-missing"
  check
  stops "llama-server has an unresolved library" "check_glibc_floor"
  says "llama-server has an unresolved library"
  [[ "${output}" == *"libmissing.so.1 => not found"* ]]
}

@test "a Linux Vulkan wheel that needs a newer glibc than its tag is refused" {
  cell linux-vulkan
  build
  echo GLIBC_2.35 > "${T}/${P}/glibc-symbol"
  check
  stops "needs glibc 2.35, above 2.34"
}

@test "a Windows Vulkan wheel without the Vulkan backend is refused" {
  fails_without windows-vulkan "${P}/ggml-vulkan.dll" "ggml-vulkan.dll"
}

@test "a Linux ROCm wheel without the HIP backend is refused" {
  fails_without linux-rocm "${P}/libggml-hip.so" "libggml-hip.so" "rocblas/library"
}

@test "a Linux ROCm wheel without the rocBLAS kernels folder is refused" {
  fails_without linux-rocm "${P}/rocblas/library/TensileLibrary_lazy_gfx90a.dat" "rocblas/library" "-name 'libggml-cpu"
}

@test "a Linux ROCm wheel with one CPU library is refused" {
  fails_without linux-rocm "${P}/libggml-cpu-b.so" "-ge 2" "LICENSE-ROCm-hsa-rocr$"
}

@test "a Linux ROCm wheel without the licence of a ROCm library is refused" {
  local name
  for name in hsa-rocr comgr hip-runtime-amd rocblas hipblas hipblaslt rocm-llvm roctracer rocprofiler-register; do
    echo "licence: ${name}"
    fails_without linux-rocm "${I}/licenses/LICENSE-ROCm-${name}" "wheel lacks the ${name} licence"
  done
}

@test "a Linux ROCm wheel with an unresolved library is refused" {
  cell linux-rocm
  build
  echo libmissing.so.1 > "${T}/${P}/ldd-missing"
  check
  stops "llama-server has an unresolved library" "check_glibc_floor"
  says "llama-server has an unresolved library"
  [[ "${output}" == *"libmissing.so.1 => not found"* ]]
}

@test "a Linux ROCm wheel that needs a newer glibc than its tag is refused" {
  cell linux-rocm
  build
  echo GLIBC_2.35 > "${T}/${P}/glibc-symbol"
  check
  stops "needs glibc 2.35, above 2.34" "assert_rocm_host_libs"
}

@test "a Linux ROCm wheel that asks the host for a seventh library is refused" {
  cell linux-rocm
  build
  echo libextra.so.1 > "${T}/${P}/host-library"
  check
  stops "expected exactly" "assert_rocm_bundle_loads"
}

@test "a Linux ROCm wheel whose bundle does not resolve its libraries is refused" {
  cell linux-rocm
  build
  echo yes > "${T}/${P}/bundle-open"
  check
  stops "the ROCm bundle does not close"
}

@test "a Linux CUDA wheel without the NVIDIA notice is refused" {
  fails_without linux-cuda "${I}/licenses/NOTICE-NVIDIA-CUDA" "wheel lacks the NVIDIA notice" "CCCL"
}

@test "a Linux CUDA wheel without the CCCL licence is refused" {
  fails_without linux-cuda "${I}/licenses/THIRD-PARTY/llama.cpp/_deps/cccl/LICENSE" "wheel lacks the CCCL licence" "sha256sum"
}

@test "a Linux CUDA wheel with an NVIDIA file that is not the file of the toolkit is refused" {
  cell linux-cuda
  build
  put "${P}/libcudart.so.12" 644 "another cudart"
  check
  stops "libcudart.so.12: FAILED" "test -f"
}

@test "a Linux CUDA wheel without the CUDA backend is refused" {
  fails_without linux-cuda "${P}/libggml-cuda.so" "libggml-cuda.so" "-name 'libggml-cpu"
}

@test "a Linux CUDA wheel with one CPU library is refused" {
  fails_without linux-cuda "${P}/libggml-cpu-b.so" "-ge 2" "ldd"
}

@test "a Linux CUDA wheel with an unresolved library is refused" {
  cell linux-cuda
  build
  echo libmissing.so.1 > "${T}/${P}/ldd-missing"
  check
  stops "llama-server has an unresolved library"
  says "llama-server has an unresolved library"
}

@test "a Windows CUDA wheel without the NVIDIA notice is refused" {
  fails_without windows-cuda "${I}/licenses/NOTICE-NVIDIA-CUDA" "wheel lacks the NVIDIA notice" "CCCL"
}

@test "a Windows CUDA wheel without the CCCL licence is refused" {
  fails_without windows-cuda "${I}/licenses/THIRD-PARTY/llama.cpp/_deps/cccl/LICENSE" "wheel lacks the CCCL licence"
}

@test "a Windows CUDA wheel without a CUDA library is refused" {
  local dll
  for dll in ggml-cuda.dll cudart64_12.dll cublas64_12.dll cublasLt64_12.dll; do
    echo "dll: ${dll}"
    fails_without windows-cuda "${P}/${dll}" "wheel lacks ${dll}"
  done
}

@test "a macOS wheel whose binary loads a library that is not beside it is refused" {
  cell macos-metal
  build
  echo libmissing.dylib > "${T}/${P}/rpath-library"
  check
  stops "llama-server loads libmissing.dylib: missing" "du -sk"
  [[ "${output}" == *"llama-server loads libmissing.dylib: missing"* ]]
}

@test "a macOS wheel with a symlink is refused" {
  cell macos-metal
  build
  printf '%s\n' '' '(_BIN_DIR / "link").symlink_to("llama-server")' >> "${T}/${PKG}/__init__.py"
  check
  stops "unexpected symlink" "du -sk"
  says "unexpected symlink"
}

@test "a wheel whose server does not print its version is refused" {
  cell linux-cpu
  SERVER=$'#!/bin/sh\nexit 3'
  build
  check
  [ "$status" -eq 3 ]
  [[ "${stderr}" == *"llama-server --version"* ]]
  [[ "${output}" != *"llama.cpp stub"* ]]
}

@test "a GPU wheel whose server cannot list its devices is refused, and a CPU wheel is not asked" {
  local name
  for name in linux-vulkan linux-rocm linux-cuda windows-vulkan windows-cuda; do
    echo "cell: ${name}"
    cell "${name}"
    SERVER=$'#!/bin/sh\ncase "$1" in\n  --version) echo "llama.cpp stub" ;;\n  --list-devices) exit 4 ;;\nesac'
    build
    check
    [ "$status" -eq 4 ]
    [[ "${stderr}" == *" --list-devices"* ]]
  done
  cell linux-cpu
  SERVER=$'#!/bin/sh\ncase "$1" in\n  --version) echo "llama.cpp stub" ;;\n  --list-devices) exit 4 ;;\nesac'
  build
  check
  passes
}
