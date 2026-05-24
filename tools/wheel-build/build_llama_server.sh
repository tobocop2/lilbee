#!/usr/bin/env bash
# Build the standalone llama.cpp `llama-server` binary for the lilbee[multi-gpu]
# sidecar fleet, from the SAME pinned llama.cpp source that build_llama_cpp.sh
# compiles for llama-cpp-python, so the sidecar binary and the in-process library
# stay version-matched. The compiled binary is copied into the
# packaging/llama-server-wheel/ package's bin/ dir for wheel-building.
#
# Reads:
#   BACKEND            cpu|vulkan|metal|cu121..cu124|rocm|sycl
#   LLAMA_CPP_VERSION  exact version (optional; defaults to the uv.lock pin)
#   TARGET_ARCH        cross-compile target (optional; defaults to host)
#   LLAMA_BUILD_DIR    work dir (default /tmp/llama-build)

set -euxo pipefail

backend="${BACKEND:?BACKEND is required}"
build_dir="${LLAMA_BUILD_DIR:-/tmp/llama-build}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
target_arch="${TARGET_ARCH:-}"
pkg_bin_dir="${script_dir}/../../packaging/llama-server-wheel/lilbee_llama_server/bin"

version="${LLAMA_CPP_VERSION:-}"
if [ -z "${version}" ]; then
  lock_file="${script_dir}/../../uv.lock"
  version=$(grep -A1 '^name = "llama-cpp-python"$' "${lock_file}" | grep '^version = ' | head -1 | cut -d'"' -f2)
  [ -n "${version}" ] || { echo "llama-cpp-python not found in ${lock_file}" >&2; exit 1; }
fi

# llama-cpp-python vendors llama.cpp as a submodule; clone at the matching tag so
# the server is byte-compatible with the in-process library's GGUF support.
src="${build_dir}/llama-cpp-python-${version}"
mkdir -p "${build_dir}"
if [ ! -d "${src}" ]; then
  git clone --depth 1 --branch "v${version}" --recurse-submodules \
    https://github.com/abetlen/llama-cpp-python "${src}"
fi

# Same backend flags as the wheel build (GGML_* cmake flags apply to the server
# target verbatim), plus the server target itself. SSL/CURL off: the fleet only
# talks to localhost sidecars, so we avoid the OpenSSL/libcurl link deps (the
# reason cmake_args.sh disables OpenSSL find on the Intel-Mac wheel cell).
eval "$(BACKEND="${backend}" TARGET_ARCH="${target_arch}" "${script_dir}/cmake_args.sh")"
# shellcheck disable=SC2086
cmake -S "${src}/vendor/llama.cpp" -B "${src}/server-build" \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_SERVER_SSL=OFF -DLLAMA_CURL=OFF ${CMAKE_ARGS}
cmake --build "${src}/server-build" --target llama-server --config Release -j

binary=$(find "${src}/server-build" -type f \( -name 'llama-server' -o -name 'llama-server.exe' \) | head -1)
[ -n "${binary}" ] || { echo "llama-server binary not found after build" >&2; exit 1; }
mkdir -p "${pkg_bin_dir}"
cp "${binary}" "${pkg_bin_dir}/"
echo "Built llama-server (${backend}) -> ${pkg_bin_dir}/$(basename "${binary}")"
