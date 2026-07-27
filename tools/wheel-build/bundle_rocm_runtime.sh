#!/usr/bin/env bash
# Pack the ROCm userspace the engine links into the wheel's bin/ directory, so an
# AMD user needs a kernel driver and nothing else, exactly as an NVIDIA one does.
#
# Its own script rather than a block inside build_llama_server.sh because it is the
# one piece of that build that can be exercised without a GPU, a ROCm install, or a
# six-hour compile: give it a directory of ELF files and a ROCm tree and it either
# closes the closure or fails. tests/test_bundle_rocm_runtime.py does exactly that.
#
# Reads:
#   ROCM_PATH  the ROCm install to pack from (default /opt/rocm-7.2.0)
# Argument:
#   the wheel's bin/ directory, holding the freshly built engine

set -euo pipefail

pkg_bin_dir="${1:?the wheel bin/ directory is required}"
rocm_root="${ROCM_PATH:-/opt/rocm-7.2.0}"

# Two directories, because ROCm 7 splits what the engine links: hip, rocblas and
# hipblas under lib, and the clang OpenMP runtime under lib/llvm/lib. The rocm cell
# compiles with AMD's clang, so ggml's OpenMP is libomp from that second directory
# and not the host's libgomp.
rocm_lib_dirs="${rocm_root}/lib ${rocm_root}/lib/llvm/lib"
for dir in ${rocm_lib_dirs}; do
  [ -d "${dir}" ] || { echo "ROCm library dir ${dir} does not exist" >&2; exit 1; }
done

needed_by() {
  readelf -d "$1" 2>/dev/null | sed -n 's/.*NEEDED.*\[\(.*\)\]/\1/p'
}

compgen -G "${pkg_bin_dir}/libggml-hip.so*" >/dev/null || {
  echo "no libggml-hip.so was built: the rocm cell produced no backend" >&2
  exit 1
}

# Breadth-first over DT_NEEDED, seeded with everything the build produced rather than
# the HIP backend alone: llama-server and the other ggml libraries went through the
# same compiler and carry the same runtime dependencies. The closure is DISCOVERED,
# not listed, because SONAMEs move between ROCm releases (libamdhip64.so.7 on 7.2)
# and the libraries pull in each other, so a hardcoded list bakes in a version and
# misses a level.
pending=$(find "${pkg_bin_dir}" -maxdepth 1 -type f)
bundled=""
while [ -n "${pending}" ]; do
  next=""
  for lib in ${pending}; do
    for need in $(needed_by "${lib}"); do
      case " ${bundled} " in *" ${need} "*) continue ;; esac
      if [ -e "${pkg_bin_dir}/${need}" ]; then continue; fi
      # Resolve against the ROCm tree only. A miss means the host owns it.
      src=""
      for dir in ${rocm_lib_dirs}; do
        if [ -e "${dir}/${need}" ]; then src="${dir}/${need}"; break; fi
      done
      [ -n "${src}" ] || continue
      cp -L "${src}" "${pkg_bin_dir}/"
      bundled="${bundled} ${need}"
      next="${next} ${pkg_bin_dir}/${need}"
    done
  done
  pending="${next}"
done
echo "bundled ROCm runtime:${bundled:- NONE}"

# rocBLAS loads Tensile kernels as data files at runtime rather than linking them,
# and looks beside its own .so for rocblas/library, so that directory travels too and
# no environment variable is needed to find it. Forgetting it yields a library that
# loads and then fails on the first matrix multiply, a worse failure than not loading
# at all. Sized by the gfx targets built, so this tracks AMDGPU_TARGETS rather than
# being a fixed cost.
if [ -d "${rocm_root}/lib/rocblas/library" ]; then
  mkdir -p "${pkg_bin_dir}/rocblas"
  cp -RL "${rocm_root}/lib/rocblas/library" "${pkg_bin_dir}/rocblas/"
fi

# Same reasoning as the CUDA gate: a driverless runner cannot exec this to find out,
# so gate on the copy and fail the build rather than a user's machine.
for required in libamdhip64 librocblas libhipblas; do
  compgen -G "${pkg_bin_dir}/${required}.so*" >/dev/null || {
    echo "no ${required} under ${rocm_root}: the ROCm bundle is incomplete" >&2
    exit 1
  }
done
[ -d "${pkg_bin_dir}/rocblas/library" ] || {
  echo "rocBLAS kernels missing: librocblas loads them at runtime and would fail on first use" >&2
  exit 1
}

# Whether the closure actually CLOSES is not asserted here. Every machine that can
# build ROCm has ROCm installed, so any check run beside this build resolves against
# the runner's /opt/rocm and passes on a bundle that would fail on a user's machine.
# The rocm cell answers it the only way that means anything, by running the bundle
# through a real ld.so inside a container with no ROCm in it. See build-multigpu.yml.
echo "rocm bundle size: $(du -sh "${pkg_bin_dir}" | cut -f1)"
