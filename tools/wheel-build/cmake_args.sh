#!/usr/bin/env bash
# Emit the CMAKE_ARGS string for the llama-server engine source build.
#
# Single source of truth for per-OS / per-backend compile flags. Everything
# that builds the engine (build_llama_server.sh via the bundle-llama-server
# action) shells out here so no two builds end up with mismatched options.
#
# Usage:
#   eval "$(BACKEND=vulkan RUNNER_OS=Linux tools/wheel-build/cmake_args.sh)"
# Sets CMAKE_ARGS in the caller's shell.
#
# Or:
#   CMAKE_ARGS="$(BACKEND=vulkan RUNNER_OS=Linux tools/wheel-build/cmake_args.sh --print)"

set -euo pipefail

backend="${BACKEND:?BACKEND is required (cpu|vulkan|metal|cu121|cu122|cu123|cu124|cu125|rocm|sycl)}"
runner_os="${RUNNER_OS:-$(uname -s)}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${script_dir}/../../engine-versions.env"

# TARGET_ARCH: cross-compile target. Defaults to host arch.
target_arch="${TARGET_ARCH:-$(uname -m)}"
case "${target_arch}" in
  arm64|aarch64) target_arch=arm64 ;;
  x86_64|amd64)  target_arch=x86_64 ;;
esac

# Universal portability flags applied to every x86_64 build:
#
#   GGML_NATIVE=OFF — no -march=native; builds done on AVX2-capable
#                     runners would otherwise crash with "Illegal
#                     instruction" on weaker pool members.
#
# GPU cells cap the single libggml-cpu.so at AVX baseline (Sandy Bridge
# 2011+); each extension is explicitly OFF because ggml's CMake
# auto-enables whatever the build host supports.
# The macOS x86_64 CPU cell builds runtime dispatch instead: one
# libggml-cpu-<variant>.so per x86 feature level, loaded best-match at
# backend init. GGML_BACKEND_DL would split GPU backends into DL modules
# too, so the GPU cells stay single-variant.
# arm64: NEON is mandatory in ARMv8; one variant covers all aarch64.
# LLAMA_BUILD_UI=OFF + LLAMA_USE_PREBUILT_UI=OFF: no npm UI build, no HF
# asset download (the pinned bucket revision is pruned upstream); lilbee
# never serves the server web UI.
ui_off="-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF"
common_x86="${ui_off} -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF -DGGML_AVX512=OFF"
dispatch_x86="${ui_off} -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON"
common_arm="${ui_off} -DGGML_NATIVE=OFF"

# Every AMD target lilbee wants to ship for: gfx906 (MI50), gfx908 (MI100),
# gfx90a (MI200), gfx942 (MI300), gfx950 (MI350), gfx1030 (RDNA2), gfx1100/1101/1102
# (RDNA3), gfx1150/1151 (RDNA3.5 APUs), gfx1200/1201 (RDNA4). Intent, not support:
# targets the installed ROCm cannot serve are filtered below, not removed here.
rocm_wanted_targets="gfx906 gfx908 gfx90a gfx942 gfx950 gfx1030 gfx1100 gfx1101 gfx1102 gfx1150 gfx1151 gfx1200 gfx1201"

# The subset of those the ROCm at $1 actually supports, as a cmake list: the
# intersection of the device bitcode (what clang can compile; one unsupported
# target fails the whole configure) and the rocBLAS lazy Tensile masters (what
# the runtime can execute; a target with bitcode but no masters builds fine and
# aborts inside rocBLAS at the first batched GEMM, as 7.2's gfx906 does).
rocm_buildable_targets() {
  local root="$1" arch targets="" no_bitcode="" no_kernels=""
  local kernels_dir="${root}/lib/rocblas/library" check_kernels="1"
  if [ ! -d "${kernels_dir}" ]; then
    check_kernels=""
    echo "cmake_args.sh: no rocBLAS kernel library under ${kernels_dir}," \
      "filtering on device bitcode alone" >&2
  fi
  for arch in ${rocm_wanted_targets}; do
    if [ ! -e "${root}/amdgcn/bitcode/oclc_isa_version_${arch#gfx}.bc" ]; then
      no_bitcode="${no_bitcode} ${arch}"
      continue
    fi
    if [ -n "${check_kernels}" ] && [ ! -e "${kernels_dir}/TensileLibrary_lazy_${arch}.dat" ]; then
      no_kernels="${no_kernels} ${arch}"
      continue
    fi
    targets="${targets}${targets:+;}${arch}"
  done

  [ -z "${no_bitcode}" ] || echo "cmake_args.sh: ROCm at ${root} has no device bitcode for:${no_bitcode}" >&2
  [ -z "${no_kernels}" ] || echo "cmake_args.sh: rocBLAS at ${root} has no GEMM kernels for:${no_kernels}" >&2

  # An install this script does not understand. Build everything rather than
  # silently emitting a wheel for no card at all.
  if [ -z "${targets}" ]; then
    echo "cmake_args.sh: no wanted target survives the filter," \
      "building every wanted target unfiltered" >&2
    echo "${rocm_wanted_targets}" | tr ' ' ';'
    return
  fi
  printf '%s' "${targets}"
}

case "${backend}_${runner_os}" in
  cpu_Linux)
    args="${common_x86} -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_VULKAN=OFF"
    ;;
  cpu_macOS)
    if [ "${target_arch}" = "x86_64" ]; then
      # Disable OpenSSL find_package: arm64 runner has no x86_64 libssl,
      # so cpp-httplib's TLS code fails to link. We don't ship llama-server.
      args="${dispatch_x86} -DCMAKE_OSX_ARCHITECTURES=x86_64 -DGGML_METAL=OFF -DGGML_BLAS=OFF -DCMAKE_DISABLE_FIND_PACKAGE_OpenSSL=ON"
    else
      args="${common_arm} -DGGML_METAL=OFF -DGGML_BLAS=OFF"
    fi
    ;;
  cpu_Windows)
    args="${common_x86} -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_BLAS=OFF"
    ;;
  metal_macOS)
    if [ "${target_arch}" = "x86_64" ]; then
      echo "metal backend is arm64-only; use cpu for Intel Mac" >&2
      exit 1
    fi
    args="${common_arm} -DGGML_METAL=ON -DGGML_BLAS=OFF"
    ;;
  vulkan_Linux|vulkan_Windows)
    # Vulkan is cross-vendor (NVIDIA, AMD, Intel Arc). Single Vulkan-built
    # wheel covers the GPU users on Linux/Windows. CUDA stays off here — a
    # CUDA-built wheel is a separate variant on the extra-index publish.
    args="${common_x86} -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_BLAS=OFF"
    ;;
  cu121_Linux|cu121_Windows|cu122_Linux|cu122_Windows|cu123_Linux|cu123_Windows|cu124_Linux|cu124_Windows|cu125_Linux|cu125_Windows)
    args="${common_x86} -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=all-major -DGGML_VULKAN=OFF -DGGML_BLAS=OFF"
    ;;
  rocm_Linux)
    # ROCm on Windows is preview-quality, so Linux only.
    # GGML_HIP, not GGML_HIPBLAS: cmake caches an unknown -D instead of failing, so
    # the renamed-away spelling built a CPU-only engine.
    # HIP device code needs AMD's clang, which ROCm 7 keeps under lib/llvm.
    rocm_root="${ROCM_PATH:-/opt/rocm-${ENGINE_ROCM_VERSION}}"
    rocm_clang="${rocm_root}/lib/llvm/bin/clang"
    rocm_targets="$(rocm_buildable_targets "${rocm_root}")"
    args="${common_x86} -DGGML_HIP=ON -DAMDGPU_TARGETS=${rocm_targets} -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DCMAKE_C_COMPILER=${rocm_clang} -DCMAKE_CXX_COMPILER=${rocm_clang}++"
    ;;
  sycl_Linux|sycl_Windows)
    # Intel oneAPI SYCL — Intel Arc + Data Center Max GPUs.
    args="${common_x86} -DGGML_SYCL=ON -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx"
    ;;
  metal_*|vulkan_macOS|cu*_macOS|rocm_*|sycl_macOS)
    echo "tools/wheel-build/cmake_args.sh: backend '${backend}' is not supported on ${runner_os}" >&2
    exit 1
    ;;
  *)
    echo "tools/wheel-build/cmake_args.sh: unknown backend '${backend}' on ${runner_os}" >&2
    exit 1
    ;;
esac

# Default: assignable. With --print: bare value for capture.
case "${1:-}" in
  --print)
    printf '%s\n' "${args}"
    ;;
  *)
    printf 'CMAKE_ARGS=%q\n' "${args}"
    ;;
esac
