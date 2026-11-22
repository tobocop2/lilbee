#!/usr/bin/env bash
# Build and check the ik_llama.cpp engine pack of one cell. Run it from a fresh lilbee checkout:
#
#   OUT=/path/to/out BACKEND=cpu ... bash tools/wheel-build/pack_ik.sh
#
# The engine-packs repository sets the variables from its cells file (scripts/pack.sh):
#   UPSTREAM_REPO, UPSTREAM_COMMIT   the server source; the wheel version names the first 12 digits of the commit
#   BACKEND, PLATFORM                the backend (cpu or cuda) and the platform tag of the wheel
#   BUILD_NUMBER                     the build number of the wheel
#   LICENSE_TERMS                    the licence terms of the cell, joined with " AND "
#   MANYLINUX_IMAGE                  the build image
#   OUT                              the folder that receives the checked wheel and its sha256 file
#
# The script needs Linux, Python 3.12, git, docker, ldd and objdump. The build writes into the wheel project, so the
# script refuses a checkout that an earlier build used.

set -euo pipefail

# The mode of each file goes into the wheel.
umask 022

# shellcheck source=tools/wheel-build/pack_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/pack_common.sh"

WHEEL_TREE=packaging/engine-wheel-ik-llama-cpp
PACKAGE=lilbee_engine_ik_llama_cpp
ADAPTER=src/lilbee/providers/fleet/engines/ik_llama.py

require_inputs() {
  local os name
  os="$(host_os)"
  [ "${os}" = Linux ] || { echo "pack_ik: no cell builds on ${os}" >&2; return 2; }
  for name in UPSTREAM_REPO UPSTREAM_COMMIT BACKEND PLATFORM BUILD_NUMBER LICENSE_TERMS MANYLINUX_IMAGE OUT; do
    [ -n "${!name:-}" ] || { echo "pack_ik: ${name} is not set" >&2; return 2; }
  done
  for name in git docker ldd objdump; do
    command -v "${name}" > /dev/null || { echo "pack_ik: ${name} is not installed" >&2; return 2; }
  done
  for name in ik-src cuda-libs.sha256 cuda-stub vendored-terms.txt wheel pack-venv check-venv \
    "${WHEEL_TREE}/${PACKAGE}/bin" "${WHEEL_TREE}/dist"; do
    [ ! -e "${name}" ] || { echo "pack_ik: an earlier build used this checkout (${name} exists). Use a fresh one." >&2; return 2; }
  done
}

# Fetch the commit and build the server in the manylinux image. A CUDA build also copies the NVIDIA files unchanged
# and writes cuda-libs.sha256 and NOTICE-NVIDIA-CUDA. ik_llama_manylinux.sh installs the pinned build tools there.
build_server() {
  mkdir -p "${WHEEL_TREE}/${PACKAGE}/bin"
  docker run --rm -v "${PWD}:/work" -w /work \
    -e UPSTREAM_REPO -e UPSTREAM_COMMIT -e BACKEND \
    -e "BIN_DIR=/work/${WHEEL_TREE}/${PACKAGE}/bin" \
    -e SRC_DIR=/work/ik-src \
    -e SHA_FILE=/work/cuda-libs.sha256 \
    -e "NOTICE_FILE=/work/${WHEEL_TREE}/NOTICE-NVIDIA-CUDA" \
    -e "OWNER=$(id -u):$(id -g)" \
    "${MANYLINUX_IMAGE}" \
    bash /work/tools/wheel-build/ik_llama_manylinux.sh
  ls -l "${WHEEL_TREE}/${PACKAGE}/bin"
  if [ "${BACKEND}" = cuda ]; then cat cuda-libs.sha256; fi
  df -h / | tail -1
}

# Collect the licences of the server and the code it carries, and set LICENSE_EXPRESSION.
collect_licences() {
  local lic="${PWD}/${WHEEL_TREE}" file newline=$'\n'
  # The locale sets the order of the licence terms and of the list of the copied files, which are in the wheel.
  local -x LC_ALL=C
  cp ik-src/LICENSE "${lic}/LICENSE-ik_llama.cpp"
  # The server compiles in two web pages and the sqlite_modern_cpp headers, all under skipped folders.
  bash tools/wheel-build/collect_vendored_licences.sh \
    --ship examples/server/public/index.html.gz \
    --ship examples/server/public_llamacpp/index_llamacpp.html.gz \
    --ship examples/server/sqlite_modern_cpp \
    ik-src "${lic}/THIRD-PARTY/ik_llama.cpp" vendored-terms.txt
  # The wheel states every licence it carries: the terms of the cell, then the vendored terms.
  LICENSE_EXPRESSION="$({ printf '%s\n' "${LICENSE_TERMS// AND /${newline}}"; tr -d '\r' < vendored-terms.txt | sort -u; } | merge_terms)"
  echo "License-Expression: ${LICENSE_EXPRESSION}"
  for file in "${lic}"/LICENSE-*; do
    [ -s "${file}" ] || { echo "pack_ik: empty licence file ${file}" >&2; return 1; }
  done
  find "${lic}/THIRD-PARTY" -type f | wc -l
}

# The subshell keeps the build variables away from the checks.
build_wheel() (
  # The wheel build and the retag stamp every entry with the commit time of the source.
  epoch="$(bash tools/wheel-build/source_date_epoch.sh ik-src)"
  export "${epoch?}"
  python tools/wheel-build/stamp_engine_wheel.py "${WHEEL_TREE}/pyproject.toml" "0+git${UPSTREAM_COMMIT:0:12}" "${LICENSE_EXPRESSION}"
  cd "${WHEEL_TREE}"
  install_build_tools python ../../tools/wheel-build/build-tools.constraints
  grep -E '^(version|license|requires)' pyproject.toml
  build_wheel_with_tools python -C--build-option="--build-number=${BUILD_NUMBER}.${BACKEND}"
  python -m wheel tags --remove --platform-tag "${PLATFORM}" dist/*.whl
  ls dist
)

check_tag() {  # wheel
  local name info
  name="$(basename "$1")"
  [ "${name}" = "${PACKAGE}-0+git${UPSTREAM_COMMIT:0:12}-${BUILD_NUMBER}.${BACKEND}-py3-none-${PLATFORM}.whl" ] \
    || { echo "pack_ik: wrong name ${name}: expected version 0+git${UPSTREAM_COMMIT:0:12}, build ${BUILD_NUMBER}.${BACKEND} and platform ${PLATFORM}" >&2; return 1; }
  info="$(dist_info "$1" WHEEL)" || return 1
  grep -qxF "Tag: py3-none-${PLATFORM}" <<< "${info}" \
    || { echo "pack_ik: the WHEEL file of ${name} lacks the tag py3-none-${PLATFORM}" >&2; return 1; }
}

# The server links the driver library, which a build host lacks. The toolkit stub stands in for it.
fetch_driver_stub() {
  mkdir cuda-stub
  docker run --rm -v "${PWD}/cuda-stub:/stub" \
    "${MANYLINUX_IMAGE}" \
    bash -euo pipefail -c 'curl --retry 5 --retry-all-errors -fsSL -o /etc/yum.repos.d/cuda-rhel9.repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo && dnf install -y cuda-driver-devel-12-4-12.4.127-1 && cp -L /usr/local/cuda-12.4/targets/x86_64-linux/lib/stubs/libcuda.so /stub/libcuda.so.1'
  ls -l cuda-stub
}

# Install the wheel, check the libraries of its server and start it. The subshell keeps the library path here.
start_server() (  # wheel
  root="${PWD}"
  python -m venv check-venv
  check-venv/bin/python -m pip install -q "$1"
  exe="$(check-venv/bin/python -c "import ${PACKAGE} as m, pathlib; print(pathlib.Path(m.__file__).parent / 'bin' / 'llama-server')")"
  echo "llama-server: ${exe}"
  bindir="$(dirname "${exe}")"
  ls -l "${bindir}"
  # A library path of this computer must not hide a library that the pack lacks.
  unset LD_LIBRARY_PATH
  if [ "${BACKEND}" = cuda ]; then
    # The NVIDIA files must match the files of the toolkit byte for byte (EULA 2.3).
    (cd "${bindir}" && sha256sum -c "${root}/cuda-libs.sha256")
    export LD_LIBRARY_PATH="${root}/cuda-stub"
  fi
  libraries="$(ldd "${exe}")"
  if grep 'not found' <<< "${libraries}"; then echo "pack_ik: llama-server has an unresolved library" >&2; exit 1; fi
  bash tools/wheel-build/check_glibc_floor.sh "${bindir}"
  cd /
  "${exe}" --version
)

main() {
  local wheel
  cd "$(dirname "${BASH_SOURCE[0]}")/../.."
  require_inputs
  use_build_python
  # A checkout that another umask made holds other modes.
  chmod -R u=rwX,go=rX "${WHEEL_TREE}"
  build_server
  collect_licences
  build_wheel
  collect_wheel "${WHEEL_TREE}/dist" wheel
  wheel="$(ls wheel/*.whl)"
  check_tag "${wheel}"
  python tools/wheel-build/check_ik_wheel.py "${wheel}" "${BACKEND}" "${LICENSE_EXPRESSION}" "${UPSTREAM_COMMIT}" "${ADAPTER}"
  if [ "${BACKEND}" = cuda ]; then fetch_driver_stub; fi
  start_server "${wheel}"
  # Only a wheel that passed its checks reaches the output folder.
  mkdir -p "${OUT}"
  cp wheel/* "${OUT}/"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main
fi
