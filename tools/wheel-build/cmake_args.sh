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

backend="${BACKEND:?BACKEND is required (cpu|vulkan|metal|cu121|cu122|cu123|cu124|cu125|rocm|sycl)}"
runner_os="${RUNNER_OS:-$(uname -s)}"

# TARGET_ARCH lets a macOS arm64 host cross-compile an x86_64 wheel for
# Intel Macs (no free GitHub-hosted Intel runner exists since macos-13
# was retired). Default: host architecture.
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
# Why we don't use GGML_CPU_ALL_VARIANTS + GGML_BACKEND_DL:
#   The DL-based runtime variant dispatch builds correctly but
#   llama-cpp-python's Python binding does not call
#   ggml_backend_load_all_from_path at startup, so no CPU backend is
#   registered at runtime and Llama(model_path=...) fails with
#   "Failed to load model from file" (no compute device available).
#   Confirmed in b443. The DL split is a llama.cpp-only mechanism;
#   adopting it requires a llama-cpp-python upstream change.
#
# Instead: cap the single libggml-cpu.so at AVX baseline (Sandy Bridge
# 2011+). Explicitly disable AVX2 / FMA / F16C / BMI2 / AVX512 / VNNI
# so ggml's CMake (which auto-enables those when GGML_NATIVE=OFF on a
# build host that supports them) doesn't bake them in. The resulting
# wheel runs on any x86_64 CPU from 2011 forward — a Sandy Bridge Xeon
# E5-2609 included.
#
# Modern users (Haswell+) lose ~10–20% CPU perf vs. an AVX2-tuned wheel,
# but they're typically on GPU paths (Vulkan / CUDA) for chat and only
# hit CPU for embedding fallback, where the loss is negligible. Users
# who want the absolute fastest CPU path can install from the per-CUDA
# extra-index — those wheels target a specific GPU and don't care about
# CPU portability.
#
# arm64: NEON is mandatory in ARMv8 so a single baseline variant covers
# every aarch64 system.
common_x86="-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF -DGGML_AVX512=OFF"
common_arm="-DGGML_NATIVE=OFF"

case "${backend}_${runner_os}" in
  cpu_Linux)
    args="${common_x86} -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_VULKAN=OFF"
    ;;
  cpu_macOS)
    if [ "${target_arch}" = "x86_64" ]; then
      # Intel Mac wheel cross-compiled from arm64 host. AVX baseline
      # (Sandy Bridge+) so the wheel runs on every Intel Mac that can
      # boot macOS 11. CMAKE_OSX_ARCHITECTURES forces ggml's CMake to
      # emit x86_64 object code instead of inheriting the host arch.
      args="${common_x86} -DCMAKE_OSX_ARCHITECTURES=x86_64 -DGGML_METAL=OFF -DGGML_BLAS=OFF"
    else
      args="${common_arm} -DGGML_METAL=OFF -DGGML_BLAS=OFF"
    fi
    ;;
  cpu_Windows)
    args="${common_x86} -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_BLAS=OFF"
    ;;
  metal_macOS)
    if [ "${target_arch}" = "x86_64" ]; then
      echo "tools/wheel-build/cmake_args.sh: metal backend not shipped for Intel Mac (use cpu)" >&2
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
