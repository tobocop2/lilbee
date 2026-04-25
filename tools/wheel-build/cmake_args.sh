#!/usr/bin/env bash
# Emit the CMAKE_ARGS string for an llama-cpp-python source build.
#
# Single source of truth for per-OS / per-backend compile flags. Both
# build-wheels.yml and release.yml shell out here so there's no chance of
# the wheel and the frozen exe ending up with mismatched compile options.
#
# Usage:
#   eval "$(BACKEND=vulkan RUNNER_OS=Linux tools/wheel-build/cmake_args.sh)"
# Sets CMAKE_ARGS in the caller's shell.
#
# Or:
#   CMAKE_ARGS="$(BACKEND=vulkan RUNNER_OS=Linux tools/wheel-build/cmake_args.sh --print)"

set -euo pipefail

backend="${BACKEND:?BACKEND is required (cpu|vulkan|metal|cu121|cu122|cu123|cu124|rocm|sycl)}"
runner_os="${RUNNER_OS:-$(uname -s)}"

# Universal portability flags applied to every x86_64 build:
#
#   GGML_NATIVE=OFF       — no -march=native; builds done on AVX2-capable
#                           runners would otherwise crash with "Illegal
#                           instruction" on weaker pool members.
#   GGML_CPU_ALL_VARIANTS — compile every CPU backend variant (SSE4.2,
#                           AVX, AVX2, AVX2+VNNI, AVX-512). ggml's runtime
#                           backend registry picks the highest variant
#                           the host CPU actually supports. Without this,
#                           a wheel built on a CI runner with AVX2 SIGILLs
#                           on Sandy Bridge / older Xeons that only have
#                           plain AVX. With it, one wheel works on every
#                           x86_64 CPU from 2008 forward.
#
# CPU_ALL_VARIANTS requires GGML_BACKEND_DL=ON: it builds each CPU variant
# (sse42, sandybridge, haswell, skylake, icelake, ...) as a separately-
# loadable .so, then dlopens the highest one the host CPU can run at
# startup. Without GGML_BACKEND_DL, ALL_VARIANTS silently falls back to a
# single-variant build matching the build host (CI runner = AVX2-capable),
# which then SIGILLs on Sandy Bridge.
#
# arm64 builds drop both flags — NEON is mandatory in ARMv8 so a single
# variant covers every aarch64 system.
common_x86="-DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON"
common_arm="-DGGML_NATIVE=OFF"

case "${backend}_${runner_os}" in
  cpu_Linux)
    args="${common_x86} -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_VULKAN=OFF"
    ;;
  cpu_macOS)
    args="${common_arm} -DGGML_METAL=OFF -DGGML_BLAS=OFF"
    ;;
  cpu_Windows)
    args="${common_x86} -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_BLAS=OFF"
    ;;
  metal_macOS)
    args="${common_arm} -DGGML_METAL=ON -DGGML_BLAS=OFF"
    ;;
  vulkan_Linux|vulkan_Windows)
    # Vulkan is cross-vendor (NVIDIA, AMD, Intel Arc). Single Vulkan-built
    # wheel covers the GPU users on Linux/Windows. CUDA stays off here — a
    # CUDA-built wheel is a separate variant on the extra-index publish.
    args="${common_x86} -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_BLAS=OFF"
    ;;
  cu121_Linux|cu121_Windows)
    args="${common_x86} -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=all-major -DGGML_VULKAN=OFF -DGGML_BLAS=OFF"
    ;;
  cu122_Linux|cu122_Windows|cu123_Linux|cu123_Windows|cu124_Linux|cu124_Windows|cu125_Linux|cu125_Windows)
    args="${common_x86} -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=all-major -DGGML_VULKAN=OFF -DGGML_BLAS=OFF"
    ;;
  rocm_Linux)
    # AMD ROCm/HIP. Only Linux is realistic — ROCm on Windows is preview-quality.
    # Compute capabilities cover RDNA2/RDNA3/RDNA4 + CDNA: gfx906 (MI50),
    # gfx908 (MI100), gfx90a (MI200), gfx940/942 (MI300), gfx1030 (RDNA2),
    # gfx1100 (RDNA3), gfx1101/1102 (Navi 32/33).
    args="${common_x86} -DGGML_HIPBLAS=ON -DAMDGPU_TARGETS=gfx906;gfx908;gfx90a;gfx940;gfx942;gfx1030;gfx1100;gfx1101;gfx1102 -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_BLAS=OFF"
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
