#!/usr/bin/env bash
# Check one llama.cpp engine wheel: install it, then check its files, its licences and its libraries, and start the
# server.
#
#   check_llama_wheel.sh <wheel>
#
# Run it in the checkout that pack_llama.sh built in. It reads license-expression.txt, cuda-libs.sha256 and
# vs-licence.docx there. Reads:
#   BACKEND   the backend of the wheel
#   RUNNER_OS the host as GitHub names it (optional on a computer that uname names)

set -euo pipefail

# shellcheck source=tools/wheel-build/pack_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/pack_common.sh"

root="${PWD}"
tools="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
os="$(host_os)"

refuse() {
  echo "$1" >&2
  return 1
}

# Fail when the listing of the wheel lacks a line that matches the pattern. A pattern that ends in $ matches the end of
# the name.
listed() {  # pattern message
  grep -q "$1" wheel-listing.txt || refuse "$2"
}

unresolved() {  # server
  local libs
  libs="$(ldd "$1")" || refuse "ldd failed on $1"
  if grep 'not found' <<< "${libs}"; then refuse "llama-server has an unresolved library"; fi
}

cpu_libraries() {  # bindir
  test "$(find "$1" -name 'libggml-cpu-*.so' | wc -l)" -ge 2
}

check_files() {  # bindir
  local bindir="$1" tool
  ls "${bindir}"
  for tool in llama-swap gguf-parser; do
    test -e "${bindir}/${tool}" -o -e "${bindir}/${tool}.exe" || refuse "wheel lacks ${tool}"
  done
  if [ "${os}" = Windows ]; then
    # The pack deletes each Visual C++ runtime DLL that no file imports, so a DLL is required only when a file imports it.
    python "${tools}/qa/assert_engine_bundle.py" --imports-only "${bindir}" || refuse "a file of the wheel imports a DLL that nothing provides"
  fi
}

check_licences() {  # wheel bindir
  local wheel="$1" bindir="$2" expression lic third metadata header
  expression="$(cat license-expression.txt)"
  # Only the header block of METADATA counts: its description follows the first blank line.
  metadata="$(unzip -p "${wheel}" '*.dist-info/METADATA')"
  header="$(sed '/^$/q' <<< "${metadata}")"
  grep -Fx "License-Expression: ${expression}" <<< "${header}" \
    || refuse "wheel lacks the licence expression ${expression} in its METADATA header"
  # The listing goes to a file: grep -q stops at its first match, and a pipe from unzip then fails with SIGPIPE under
  # pipefail on a long listing.
  unzip -l "${wheel}" > wheel-listing.txt
  for lic in llama.cpp llama-swap gguf-parser; do
    listed "dist-info/licenses/LICENSE-${lic}$" "wheel lacks the ${lic} licence"
  done
  for third in llama-swap gguf-parser llama.cpp; do
    listed "dist-info/licenses/THIRD-PARTY/${third}/" "wheel lacks the ${third} third-party licences"
  done
  listed "dist-info/licenses/LICENSE-Go$" "wheel lacks the Go library licence"
  if [ "${os}" = Windows ]; then
    listed "dist-info/licenses/NOTICE-MICROSOFT-VC-RUNTIME$" "wheel lacks the Visual C++ runtime notice"
    python -I "${tools}/wheel-build/vs_distributable_terms.py" --verify vs-licence.docx "${wheel}"
  fi
  if [ -f "${bindir}/libomp.dll" ]; then
    listed "dist-info/licenses/LICENSE-LLVM-OpenMP$" "wheel lacks the LLVM OpenMP licence"
  fi
}

check_backend() {  # bindir
  local bindir="$1" lic dll
  if [ "${BACKEND}" = cpu ] && [ "${os}" = Linux ]; then
    cpu_libraries "${bindir}"
    unresolved "${bindir}/llama-server"
    bash "${tools}/wheel-build/check_glibc_floor.sh" "${bindir}"
  fi
  if [ "${BACKEND}" = vulkan ]; then
    if [ "${os}" = Windows ]; then
      test -f "${bindir}/ggml-vulkan.dll"
    else
      test -f "${bindir}/libggml-vulkan.so"
      cpu_libraries "${bindir}"
      if find "${bindir}" -name 'libvulkan*' | grep .; then refuse "the Vulkan loader must stay with the host"; fi
      unresolved "${bindir}/llama-server"
      bash "${tools}/wheel-build/check_glibc_floor.sh" "${bindir}"
    fi
  fi
  if [ "${BACKEND}" = rocm ]; then
    test -f "${bindir}/libggml-hip.so"
    test -d "${bindir}/rocblas/library"
    cpu_libraries "${bindir}"
    for lic in hsa-rocr comgr hip-runtime-amd rocblas hipblas hipblaslt rocm-llvm roctracer rocprofiler-register; do
      listed "dist-info/licenses/LICENSE-ROCm-${lic}$" "wheel lacks the ${lic} licence"
    done
    unresolved "${bindir}/llama-server"
    bash "${tools}/wheel-build/check_glibc_floor.sh" "${bindir}"
    # A clean container names exactly the six host libraries; with them installed, everything resolves.
    bash "${tools}/qa/assert_rocm_host_libs.sh" "${bindir}"
    bash "${tools}/qa/assert_rocm_bundle_loads.sh" "${bindir}"
  fi
  if [ "${BACKEND}" = cuda ]; then
    listed "dist-info/licenses/NOTICE-NVIDIA-CUDA$" "wheel lacks the NVIDIA notice"
    listed "dist-info/licenses/THIRD-PARTY/llama.cpp/_deps/cccl/LICENSE$" "wheel lacks the CCCL licence"
    if [ "${os}" = Windows ]; then
      for dll in ggml-cuda.dll cudart64_12.dll cublas64_12.dll cublasLt64_12.dll; do
        test -f "${bindir}/${dll}" || refuse "wheel lacks ${dll}"
      done
    else
      # The NVIDIA files must match the toolkit's files byte for byte (EULA 2.3).
      (cd "${bindir}" && sha256sum -c "${root}/cuda-libs.sha256")
      test -f "${bindir}/libggml-cuda.so"
      cpu_libraries "${bindir}"
      unresolved "${bindir}/llama-server"
    fi
  fi
}

# Every library that a binary loads by @rpath or @loader_path must be a file beside it.
check_macos() {  # bindir
  local bindir="$1" missing=0 f name kind libs
  for f in "${bindir}"/*; do
    kind="$(file "${f}")" || refuse "file failed on ${f}"
    case "${kind}" in *Mach-O*) ;; *) continue ;; esac
    libs="$(otool -L "${f}")" || refuse "otool failed on ${f}"
    for name in $(awk 'NR > 1 && $1 ~ /^@(rpath|loader_path)\// {n = split($1, a, "/"); print a[n]}' <<< "${libs}"); do
      test -f "${bindir}/${name}" || { echo "${f##*/} loads ${name}: missing"; missing=1; }
    done
  done
  test "${missing}" -eq 0
  find "${bindir}" -type l | grep . && refuse "unexpected symlink"
  du -sk "${bindir}"
}

main() {
  local wheel exe py bindir
  if [ "$#" -ne 1 ]; then
    echo "usage: check_llama_wheel.sh <wheel>" >&2
    return 2
  fi
  wheel="$1"
  case "${BACKEND-}" in cpu | vulkan | rocm | cuda | metal) ;; *) refuse "unknown BACKEND: ${BACKEND-}" ;; esac
  # A venv of an earlier check keeps its installed copy of a wheel with the same version.
  [ ! -e check-venv ] || { echo "check_llama_wheel: an earlier check used this folder (check-venv exists). Use a fresh one." >&2; return 2; }
  python -m venv check-venv
  if [ "${os}" = Windows ]; then py=check-venv/Scripts/python.exe; else py=check-venv/bin/python; fi
  "${py}" -m pip install -q "${wheel}"
  exe="$("${py}" -c "import lilbee_engine_llama_cpp as m; print(m.get_llama_server_path().as_posix())")"
  echo "llama-server: ${exe}"
  bindir="$(dirname "${exe}")"
  check_files "${bindir}"
  check_licences "${wheel}" "${bindir}"
  check_backend "${bindir}"
  if [ "${os}" = macOS ]; then check_macos "${bindir}"; fi
  cd /
  "${exe}" --version
  case "${BACKEND}" in cuda | vulkan | rocm) "${exe}" --list-devices ;; esac
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
