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
  # spirv-headers provides <spirv/unified1/spirv.hpp> (the `spv` namespace);
  # ggml-vulkan.cpp needs it to patch SPIR-V, and apt's libvulkan-dev does
  # not pull it in. The LunarG SDK bundles it, which is why Windows builds
  # without this. Without it the build fails: "'spv' has not been declared".
  sudo apt-get update
  sudo apt-get install -y libvulkan-dev libvulkan1 glslc spirv-headers
}

linux_install_cuda() {
  # cu121 -> apt cuda-toolkit-12-1, /usr/local/cuda-12.1, etc.
  local cu="$1"
  local cu_ver="${cu#cu}"           # 121, 124, 125
  local major="${cu_ver:0:2}"       # 12
  local minor="${cu_ver:2:1}"       # 1, 4, 5
  local apt_pkg="cuda-toolkit-${major}-${minor}"
  local cuda_home="/usr/local/cuda-${major}.${minor}"

  # Pick the NVIDIA repo matching the runner's Ubuntu version. Ubuntu
  # 24.04 (noble) only carries the latest CUDA toolkit; older 12.x
  # versions live only in the ubuntu2204 repo. Detect via /etc/os-release.
  local ubuntu_codename
  ubuntu_codename=$(. /etc/os-release && echo "${VERSION_CODENAME}")
  local repo_id
  case "${ubuntu_codename}" in
    noble)  repo_id="ubuntu2404" ;;
    jammy)  repo_id="ubuntu2204" ;;
    focal)  repo_id="ubuntu2004" ;;
    *)
      echo "tools/wheel-build/install_gpu_toolkit.sh: unsupported Ubuntu codename '${ubuntu_codename}'" >&2
      exit 1
      ;;
  esac

  if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${repo_id}/x86_64/cuda-keyring_1.1-1_all.deb" \
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
  # Outside GitHub Actions (e.g. a QA pod) there is no $GITHUB_ENV/$GITHUB_PATH
  # to carry the toolkit location to the next step. Emit the same facts as a
  # sourceable file so the caller can put nvcc on PATH for the build.
  if [ -n "${TOOLKIT_ENV_FILE:-}" ]; then
    {
      echo "export PATH=\"${cuda_home}/bin:\$PATH\""
      echo "export CUDA_HOME=\"${cuda_home}\""
      echo "export CUDACXX=\"${cuda_home}/bin/nvcc\""
    } >> "$TOOLKIT_ENV_FILE"
  fi
}

linux_install_rocm() {
  # ROCm 7.2, and the version is not incidental. The engine's vendored llama.cpp
  # refuses anything below 6.1 outright, and 6.1 itself fails to compile it:
  # ggml-cuda/mma.cuh, which the HIP backend reuses, hits -Wc++11-narrowing on
  # that release's clang 17. Both were reproduced on an MI300X; 7.2 builds clean.
  #
  # The versioned package names are also not incidental. An unversioned
  # rocm-hip-sdk resolves its compiler dependency against any other ROCm already
  # on the image, which on a box that ships one installs every library and no
  # clang at all, and the build then fails looking for a compiler that was never
  # unpacked.
  if [ ! -d /opt/rocm-7.2.0 ]; then
    sudo mkdir -p --mode=0755 /etc/apt/keyrings
    wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key \
      | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
    echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2 jammy main' \
      | sudo tee /etc/apt/sources.list.d/rocm.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y rocm-hip-sdk7.2.0 rocm-llvm7.2.0 hipcc7.2.0
  fi
  # ROCm 7 moved the toolchain to lib/llvm; $ROCM_PATH/llvm/bin no longer exists.
  echo "/opt/rocm-7.2.0/bin" >> "$GITHUB_PATH"
  echo "/opt/rocm-7.2.0/lib/llvm/bin" >> "$GITHUB_PATH"
  {
    echo "ROCM_PATH=/opt/rocm-7.2.0"
    echo "HIP_PATH=/opt/rocm-7.2.0"
    echo "CMAKE_PREFIX_PATH=/opt/rocm-7.2.0"
  } >> "$GITHUB_ENV"
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
