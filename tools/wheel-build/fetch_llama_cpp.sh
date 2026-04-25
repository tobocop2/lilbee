#!/usr/bin/env bash
# Download a prebuilt llama-cpp-python wheel from abetlen's index.
#
# Use this instead of build_llama_cpp.sh for backends that abetlen already
# ships — saves CI time and avoids reinventing wheels. abetlen's coverage
# at the time of writing:
#   * cpu                 (every OS, but Linux is musl-linked → unusable)
#   * cu121..cu124        (Linux x86_64 only)
#   * metal               (macOS arm64)
#
# Anything else (vulkan, rocm, sycl, Windows CUDA) must go through
# build_llama_cpp.sh.

set -euxo pipefail

version="${LLAMA_CPP_VERSION:?LLAMA_CPP_VERSION is required}"
backend="${BACKEND:?BACKEND is required}"
runner_os="${RUNNER_OS:-$(uname -s)}"
plat_tag="${LLAMA_PLAT_TAG:?LLAMA_PLAT_TAG is required (e.g. macosx_11_0_arm64, manylinux2014_x86_64)}"
build_dir="${LLAMA_BUILD_DIR:-/tmp/llama-build}"

mkdir -p "${build_dir}"

case "${backend}" in
  cpu)    index_path="cpu" ;;
  metal)  index_path="metal" ;;
  cu121|cu122|cu123|cu124) index_path="${backend}" ;;
  *)
    echo "tools/wheel-build/fetch_llama_cpp.sh: backend '${backend}' is not on abetlen's index; use build_llama_cpp.sh" >&2
    exit 1
    ;;
esac

py="cp$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"

pip download \
  --no-deps \
  --only-binary=:all: \
  --platform="${plat_tag}" \
  --python-version="${py#cp}" \
  --index-url="https://abetlen.github.io/llama-cpp-python/whl/${index_path}/" \
  "llama-cpp-python==${version}" \
  --dest="${build_dir}"

ls -lh "${build_dir}"/llama_cpp_python-*.whl
