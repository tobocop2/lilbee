#!/usr/bin/env bash
# Build llama-cpp-python from source for the requested backend.
#
# Output goes to ${LLAMA_BUILD_DIR}/llama_cpp_python-*.whl. The lilbee
# wheel vendoring step (tools/vendor/llama_cpp.py) consumes that wheel
# via --local-wheel.
#
# Reads:
#   LLAMA_CPP_VERSION   exact version, e.g. 0.3.19
#   BACKEND             cpu|vulkan|metal|cu121..cu124|rocm|sycl
#   RUNNER_OS           Linux|macOS|Windows
#   LLAMA_BUILD_DIR     output dir (default /tmp/llama-build)
#   MACOSX_DEPLOYMENT_TARGET (macOS only) deployment target

set -euxo pipefail

version="${LLAMA_CPP_VERSION:?LLAMA_CPP_VERSION is required}"
backend="${BACKEND:?BACKEND is required}"
build_dir="${LLAMA_BUILD_DIR:-/tmp/llama-build}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
target_arch="${TARGET_ARCH:-}"

mkdir -p "${build_dir}"

# Cross-compile: ARCHFLAGS drives clang, _PYTHON_HOST_PLATFORM drives the wheel tag.
if [ -n "${target_arch}" ] && [ "$(uname -s)" = "Darwin" ] && [ "${target_arch}" != "$(uname -m)" ]; then
  export ARCHFLAGS="-arch ${target_arch}"
  export _PYTHON_HOST_PLATFORM="macosx-${MACOSX_DEPLOYMENT_TARGET:-11.0}-${target_arch}"
fi

# shellcheck source=/dev/null
eval "$(BACKEND="${backend}" TARGET_ARCH="${target_arch}" "${script_dir}/cmake_args.sh")"
export CMAKE_ARGS

echo "Building llama-cpp-python==${version} (${backend}) with CMAKE_ARGS=${CMAKE_ARGS}"

PIP_CMD="${PIP_CMD:-pip}"
${PIP_CMD} wheel "llama-cpp-python==${version}" \
  --no-deps \
  --no-binary=llama-cpp-python \
  -w "${build_dir}"

ls -lh "${build_dir}"/llama_cpp_python-*.whl
