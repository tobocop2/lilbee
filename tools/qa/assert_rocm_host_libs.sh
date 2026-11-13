#!/usr/bin/env bash
# Assert the ROCm pack asks the host for exactly six shared libraries. A container with no
# libelf, libdrm, libdrm_amdgpu, libnuma, libz or libzstd runs the dynamic loader over every
# file of the pack and must report those six, and no others, as missing.
# assert_rocm_bundle_loads.sh then installs them and checks that everything resolves.
#
# Argument:
#   the wheel's bin/ directory, holding the ROCm engine

set -euo pipefail

bundle_dir="${1:?the wheel bin/ directory is required}"
[ -d "${bundle_dir}" ] || { echo "no bundle directory at ${bundle_dir}" >&2; exit 1; }
bundle_dir="$(cd "${bundle_dir}" && pwd)"

# Base libraries only: glibc, libstdc++ and libgcc. The image has a loader and no shell.
image="gcr.io/distroless/cc-debian12"
loader=/lib64/ld-linux-x86-64.so.2
expected="libdrm.so.2
libdrm_amdgpu.so.1
libelf.so.1
libnuma.so.1
libz.so.1
libzstd.so.1"

missing=""
count=0
for file in "${bundle_dir}"/llama-server "${bundle_dir}"/*.so*; do
  [ -f "${file}" ] || continue
  count=$((count + 1))
  # ld.so --list is what ldd runs; it exits non-zero for a library it cannot find.
  listing="$(docker run --rm --entrypoint "${loader}" -v "${bundle_dir}:/bundle:ro" "${image}" \
    --list "/bundle/$(basename "${file}")" 2>&1 || true)"
  missing="${missing}$(printf '%s\n' "${listing}" | sed -n 's/^[[:space:]]*\([^ ]*\) => not found.*/\1/p')"$'\n'
done
[ "${count}" -gt 0 ] || { echo "no file to check in ${bundle_dir}" >&2; exit 1; }

found="$(printf '%s' "${missing}" | sort -u | sed '/^$/d')"
echo "files checked: ${count}"
echo "missing in a clean container:"
echo "${found}"
if [ "${found}" != "${expected}" ]; then
  echo "expected exactly:" >&2
  echo "${expected}" >&2
  exit 1
fi
echo "the pack names exactly the six host libraries"
