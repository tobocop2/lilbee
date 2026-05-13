#!/usr/bin/env bash
# Build llama-cpp-python from source for the requested backend.
#
# Output goes to ${LLAMA_BUILD_DIR}/llama_cpp_python-*.whl. The lilbee
# wheel vendoring step (tools/vendor/llama_cpp.py) consumes that wheel
# via --local-wheel.
#
# Reads:
#   LLAMA_CPP_VERSION   exact version (optional; defaults to the pin in uv.lock)
#   BACKEND             cpu|vulkan|metal|cu121..cu124|rocm|sycl
#   RUNNER_OS           Linux|macOS|Windows
#   LLAMA_BUILD_DIR     output dir (default /tmp/llama-build)
#   MACOSX_DEPLOYMENT_TARGET (macOS only) deployment target

set -euxo pipefail

backend="${BACKEND:?BACKEND is required}"
build_dir="${LLAMA_BUILD_DIR:-/tmp/llama-build}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
target_arch="${TARGET_ARCH:-}"

# uv.lock is the single source of truth for the version; LLAMA_CPP_VERSION overrides it.
version="${LLAMA_CPP_VERSION:-}"
if [ -z "${version}" ]; then
  lock_file="${script_dir}/../../uv.lock"
  version=$(grep -A1 '^name = "llama-cpp-python"$' "${lock_file}" | grep '^version = ' | head -1 | cut -d'"' -f2)
  [ -n "${version}" ] || { echo "llama-cpp-python not found in ${lock_file} and LLAMA_CPP_VERSION unset" >&2; exit 1; }
fi

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

# Locate pip. uv-managed Python doesn't put pip on PATH, so prefer the
# project venv's pip when present; fall back to PATH for setup-python
# (build-*-wheels.yml) and Homebrew/system installs (release.yml on macOS).
if [ -n "${PIP_CMD:-}" ]; then
  pip_cmd=$PIP_CMD
elif [ -x ".venv/bin/pip" ]; then
  pip_cmd=".venv/bin/pip"
elif [ -x ".venv/Scripts/pip.exe" ]; then
  pip_cmd=".venv/Scripts/pip.exe"
else
  pip_cmd="pip"
fi

"$pip_cmd" wheel "llama-cpp-python==${version}" \
  --no-deps \
  --no-binary=llama-cpp-python \
  -w "${build_dir}"

ls -lh "${build_dir}"/llama_cpp_python-*.whl
