#!/usr/bin/env bash
# Install the build-time GPU SDK on the runner for a given backend.
#
# No-op when the requested backend doesn't need a separate SDK install
# (cpu, metal). For CUDA backends this installs the matching CUDA Toolkit
# version. For Vulkan this installs the Vulkan SDK + loader.
#
# Reads BACKEND and RUNNER_OS from env. Idempotent.

set -euxo pipefail

backend="${BACKEND:?BACKEND is required}"
runner_os="${RUNNER_OS:-$(uname -s)}"

linux_install_vulkan() {
  sudo apt-get update
  sudo apt-get install -y libvulkan-dev libvulkan1 glslc
}

linux_install_cuda() {
  # cu121 -> apt cuda-toolkit-12-1, /usr/local/cuda-12.1, etc.
  local cu="$1"
  local cu_ver="${cu#cu}"           # 121, 124, 125
  local major="${cu_ver:0:2}"       # 12
  local minor="${cu_ver:2:1}"       # 1, 4, 5
  local apt_pkg="cuda-toolkit-${major}-${minor}"
  local cuda_home="/usr/local/cuda-${major}.${minor}"

  if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb" \
      -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt-get update
  fi
  sudo apt-get install -y "${apt_pkg}"

  if [ -n "${GITHUB_PATH:-}" ]; then
    echo "${cuda_home}/bin" >> "$GITHUB_PATH"
  fi
  if [ -n "${GITHUB_ENV:-}" ]; then
    {
      echo "CUDA_HOME=${cuda_home}"
      echo "CUDACXX=${cuda_home}/bin/nvcc"
    } >> "$GITHUB_ENV"
  fi
}

linux_install_rocm() {
  # ROCm 6.x repo. AMD's apt repo + key. Versions advance fast; pinning to
  # 6.1 as a current-stable target.
  if ! dpkg -s rocm-dev >/dev/null 2>&1; then
    wget -q https://repo.radeon.com/amdgpu-install/6.1.2/ubuntu/jammy/amdgpu-install_6.1.60102-1_all.deb \
      -O /tmp/amdgpu-install.deb
    sudo dpkg -i /tmp/amdgpu-install.deb
    sudo apt-get update
    sudo apt-get install -y rocm-dev hip-dev rocblas-dev
  fi
  echo "/opt/rocm/bin" >> "$GITHUB_PATH"
  echo "ROCM_PATH=/opt/rocm" >> "$GITHUB_ENV"
  echo "HIP_PATH=/opt/rocm" >> "$GITHUB_ENV"
}

linux_install_sycl() {
  # Intel oneAPI repo + base toolkit (icx/icpx + DPC++ + MKL).
  if ! dpkg -s intel-basekit >/dev/null 2>&1; then
    wget -q https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
      -O /tmp/intel.pub
    sudo apt-key add /tmp/intel.pub
    echo "deb https://apt.repos.intel.com/oneapi all main" | \
      sudo tee /etc/apt/sources.list.d/oneAPI.list
    sudo apt-get update
    sudo apt-get install -y intel-basekit
  fi
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/setvars.sh
  {
    echo "PATH=$PATH"
    echo "CMPLR_ROOT=${CMPLR_ROOT:-}"
    echo "ONEAPI_ROOT=${ONEAPI_ROOT:-}"
  } >> "$GITHUB_ENV"
}

case "${backend}_${runner_os}" in
  cpu_*|metal_macOS)
    echo "no GPU SDK required for ${backend} on ${runner_os}"
    ;;
  vulkan_Linux)
    linux_install_vulkan
    ;;
  vulkan_Windows)
    echo "Vulkan SDK on Windows is installed via the jakoch/install-vulkan-sdk-action step in CI."
    echo "This script is a no-op on Windows for vulkan; run that action instead."
    ;;
  cu121_Linux|cu122_Linux|cu123_Linux|cu124_Linux|cu125_Linux)
    linux_install_cuda "${backend}"
    ;;
  cu121_Windows|cu122_Windows|cu123_Windows|cu124_Windows|cu125_Windows)
    echo "CUDA Toolkit on Windows is installed via the Jimver/cuda-toolkit action in CI."
    echo "This script is a no-op on Windows for CUDA; run that action instead."
    ;;
  rocm_Linux)
    linux_install_rocm
    ;;
  sycl_Linux)
    linux_install_sycl
    ;;
  *)
    echo "tools/wheel-build/install_gpu_toolkit.sh: unsupported '${backend}' on '${runner_os}'" >&2
    exit 1
    ;;
esac
