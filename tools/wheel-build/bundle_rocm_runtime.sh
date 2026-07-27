#!/usr/bin/env bash
# Pack the ROCm userspace the engine links into the wheel's bin/, so an AMD user
# needs only a kernel driver. Separate script because it is the one part of the
# engine build testable without a GPU (tests/test_bundle_rocm_runtime.py). Whether
# the result loads is settled by tools/qa/assert_rocm_bundle_loads.sh.
#
# Reads:
#   ROCM_PATH     the ROCm install to pack from (default from engine-versions.env)
#   ROCM_TARGETS  the gfx targets the engine was compiled for; rocBLAS kernels for any
#                 other architecture are dropped. Unset keeps them all.
# Argument:
#   the wheel's bin/ directory, holding the freshly built engine

set -euo pipefail

pkg_bin_dir="${1:?the wheel bin/ directory is required}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${script_dir}/../../engine-versions.env"

rocm_root="${ROCM_PATH:-/opt/rocm-${ENGINE_ROCM_VERSION}}"
# SONAMEs copied out of the ROCm tree, set by copy_rocm_closure.
bundled=""
# ROCm 7 splits what the engine links: hip, rocblas and hipblas under lib, clang's
# OpenMP runtime under lib/llvm/lib. The rocm cell compiles with AMD's clang, so
# ggml's OpenMP is libomp from the second rather than the host's libgomp.
rocm_lib_dirs="${rocm_root}/lib ${rocm_root}/lib/llvm/lib"

# The libraries a required-check names. Everything else is discovered.
required_libs="libamdhip64 librocblas libhipblas"

needed_by() {
  readelf -d "$1" 2>/dev/null | sed -n 's/.*NEEDED.*\[\(.*\)\]/\1/p'
}

resolve_in_rocm() {
  local soname="$1" dir
  for dir in ${rocm_lib_dirs}; do
    if [ -e "${dir}/${soname}" ]; then
      printf '%s' "${dir}/${soname}"
      return
    fi
  done
}

# Breadth-first over DT_NEEDED, seeded with everything the build produced rather than
# the HIP backend alone: llama-server went through the same compiler and links libomp
# where the backend does not. SONAMEs move between ROCm releases, so the closure is
# discovered rather than listed.
copy_rocm_closure() {
  local pending next lib soname src
  bundled=""
  pending=$(find "${pkg_bin_dir}" -maxdepth 1 -type f)
  while [ -n "${pending}" ]; do
    next=""
    for lib in ${pending}; do
      for soname in $(needed_by "${lib}"); do
        case " ${bundled} " in *" ${soname} "*) continue ;; esac
        if [ -e "${pkg_bin_dir}/${soname}" ]; then continue; fi
        src="$(resolve_in_rocm "${soname}")"
        # A miss means the host owns it.
        [ -n "${src}" ] || continue
        cp -L "${src}" "${pkg_bin_dir}/"
        bundled="${bundled} ${soname}"
        next="${next} ${pkg_bin_dir}/${soname}"
      done
    done
    pending="${next}"
  done
  echo "bundled ROCm runtime:${bundled:- NONE}"
}

# The build dereferences SONAME symlinks into real files, so each library ships up to
# three times (437 MB each for the HIP backend), and a wheel cannot carry symlinks.
# Which name is load-bearing is derived, not assumed: if anything DT_NEEDEDs a member
# of the group the backend is linked, so keep only those names; if nothing does, ggml
# dlopens the plain name, so keep that. Only drops a duplicate of a surviving name.
drop_redundant_copies() {
  local needed="" obj name stem keep
  for obj in "${pkg_bin_dir}"/*; do
    [ -f "${obj}" ] || continue
    needed="${needed} $(needed_by "${obj}" | tr '\n' ' ')"
  done

  for obj in "${pkg_bin_dir}"/*.so*; do
    [ -f "${obj}" ] || continue
    name="$(basename "${obj}")"
    stem="${name%%.so*}"
    # A name something links against is load-bearing whatever else is true.
    case " ${needed} " in *" ${name} "*) continue ;; esac
    if group_is_linked "${stem}" "${needed}"; then
      # Linked group: the SONAME carries it, so this alias is dead.
      keep=""
    else
      # dlopen group: the plain name is the only one ggml can ask for.
      [ "${name}" = "${stem}.so" ] && keep="1" || keep=""
    fi
    [ -z "${keep}" ] || continue
    identical_to_a_kept_name "${obj}" "${stem}" "${needed}" || continue
    rm -f "${obj}"
    echo "dropped redundant copy: ${name}"
  done
}

# Does anything link against some member of this library's name group?
group_is_linked() {
  local stem="$1" needed="$2" n
  for n in ${needed}; do
    case "${n}" in "${stem}.so"*) return 0 ;; esac
  done
  return 1
}

# Only delete a file whose bytes survive under another name, so a mistake in the rules
# above cannot remove the last copy of a library.
identical_to_a_kept_name() {
  local obj="$1" stem="$2" needed="$3" other name
  for other in "${pkg_bin_dir}/${stem}".so*; do
    [ -f "${other}" ] || continue
    [ "${other}" != "${obj}" ] || continue
    name="$(basename "${other}")"
    case " ${needed} " in *" ${name} "*) cmp -s "${obj}" "${other}" && return 0 ;; esac
    [ "${name}" = "${stem}.so" ] || continue
    group_is_linked "${stem}" "${needed}" || { cmp -s "${obj}" "${other}" && return 0; }
  done
  return 1
}

# A copied library keeps the runpath it was built with, which points into the ROCm
# install that will not exist on a user's machine. Repoint each at its own directory
# so the bundle resolves against itself. patchelf is what auditwheel uses for this.
repoint_runpaths() {
  local soname
  command -v patchelf >/dev/null || {
    echo "patchelf is required to relocate the bundled ROCm libraries" >&2
    exit 1
  }
  for soname in $1; do
    patchelf --set-rpath '$ORIGIN' "${pkg_bin_dir}/${soname}"
  done
}

# rocBLAS loads its Tensile kernels as data rather than linking them, and looks beside
# its own .so for them. Omitting them yields a library that loads and then fails on
# the first matrix multiply. Sized by the gfx targets built, so this tracks
# AMDGPU_TARGETS rather than being a fixed cost.
copy_rocblas_kernels() {
  [ -d "${rocm_root}/lib/rocblas/library" ] || return 0
  mkdir -p "${pkg_bin_dir}/rocblas"
  cp -RL "${rocm_root}/lib/rocblas/library" "${pkg_bin_dir}/rocblas/"
  drop_kernels_for_unbuilt_targets
}

# rocBLAS ships kernels for every architecture its build covered, which is more than the
# engine was compiled for. A kernel file names its target, so the ones for architectures
# this wheel cannot run on are dead weight. ROCM_TARGETS is what cmake was given; without
# it nothing is dropped, since guessing wrong here costs a GPU that silently falls back.
drop_kernels_for_unbuilt_targets() {
  local dir="${pkg_bin_dir}/rocblas/library" file name target keep
  [ -n "${ROCM_TARGETS:-}" ] || { echo "ROCM_TARGETS unset: keeping every rocBLAS kernel"; return 0; }
  local wanted=" $(echo "${ROCM_TARGETS}" | tr ';' ' ') "

  for file in "${dir}"/*; do
    [ -f "${file}" ] || continue
    name="$(basename "${file}")"
    # The architecture this file is for, if it names one. Files that name none are
    # shared metadata and always stay.
    # || true: grep exits 1 on the metadata files that name no architecture, which
    # set -e would otherwise treat as a build failure.
    target="$(echo "${name}" | grep -o 'gfx[0-9a-f]*' | head -1 || true)"
    [ -n "${target}" ] || continue
    case "${wanted}" in *" ${target} "*) keep="1" ;; *) keep="" ;; esac
    [ -n "${keep}" ] || rm -f "${file}"
  done
  echo "rocBLAS kernels kept for:${ROCM_TARGETS}"
}

assert_rocm_dirs_exist() {
  local dir
  for dir in ${rocm_lib_dirs}; do
    [ -d "${dir}" ] || { echo "ROCm library dir ${dir} does not exist" >&2; exit 1; }
  done
}

assert_backend_was_built() {
  compgen -G "${pkg_bin_dir}/libggml-hip.so*" >/dev/null || {
    echo "no libggml-hip.so was built: the rocm cell produced no backend" >&2
    exit 1
  }
}

# A driverless runner cannot exec this to find out, so gate on the copy.
assert_bundle_is_complete() {
  local required
  for required in ${required_libs}; do
    compgen -G "${pkg_bin_dir}/${required}.so*" >/dev/null || {
      echo "no ${required} under ${rocm_root}: the ROCm bundle is incomplete" >&2
      exit 1
    }
  done
  [ -d "${pkg_bin_dir}/rocblas/library" ] || {
    echo "rocBLAS kernels missing: librocblas loads them at runtime and would fail on first use" >&2
    exit 1
  }
}

assert_rocm_dirs_exist
assert_backend_was_built
copy_rocm_closure
drop_redundant_copies
repoint_runpaths "${bundled}"
copy_rocblas_kernels
assert_bundle_is_complete
echo "rocm bundle size: $(du -sh "${pkg_bin_dir}" | cut -f1)"
