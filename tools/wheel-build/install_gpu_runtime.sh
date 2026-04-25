#!/usr/bin/env bash
# Install only the runtime loader needed to import a GPU-built llama_cpp.
#
# Used on the verify-pypi runner (which doesn't build anything but does
# import the published wheel) and on any developer box that wants to
# `lilbee self-check` against a CPU/GPU wheel without a GPU driver
# installed.
#
# CPU-only headless boxes (cloud servers, containers, bare WSL) will fail
# to import a Vulkan-built wheel without libvulkan1 present. This script
# closes that gap so the verify lane reflects what driver-equipped users
# actually see.

set -euxo pipefail

backend="${BACKEND:?BACKEND is required}"
runner_os="${RUNNER_OS:-$(uname -s)}"

case "${backend}_${runner_os}" in
  cpu_*|metal_macOS)
    echo "no runtime loader required for ${backend} on ${runner_os}"
    ;;
  vulkan_Linux)
    sudo apt-get update
    sudo apt-get install -y libvulkan1
    ;;
  vulkan_Windows)
    echo "Vulkan runtime on Windows is installed by jakoch/install-vulkan-sdk-action with install_runtime: true."
    ;;
  cu121_Linux|cu122_Linux|cu123_Linux|cu124_Linux|cu125_Linux)
    sudo apt-get update
    sudo apt-get install -y "cuda-compat-${backend#cu}" || true
    ;;
  cu121_Windows|cu122_Windows|cu123_Windows|cu124_Windows|cu125_Windows)
    echo "CUDA runtime on Windows is provided by the Jimver/cuda-toolkit action; nothing to install here."
    ;;
  rocm_Linux)
    sudo apt-get update
    sudo apt-get install -y rocm-dev || true
    ;;
  sycl_Linux)
    sudo apt-get update
    sudo apt-get install -y intel-oneapi-runtime-libs || true
    ;;
  *)
    echo "tools/wheel-build/install_gpu_runtime.sh: unsupported '${backend}' on '${runner_os}'" >&2
    exit 1
    ;;
esac
