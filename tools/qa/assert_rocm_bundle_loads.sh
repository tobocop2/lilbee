#!/usr/bin/env bash
# Assert the bundled ROCm engine resolves every library on a machine with no ROCm.
# The build host is never that machine, so a container is. No LD_LIBRARY_PATH: the
# baked $ORIGIN runpath is what a user gets.
#
# Argument:
#   the wheel's bin/ directory, holding the bundled engine

set -euo pipefail

bundle_dir="${1:?the wheel bin/ directory is required}"

# Matches the runner the wheels are built on, so glibc is the one they were linked to.
image="ubuntu:22.04"

# The wheel's host contract: what must match the running kernel driver rather than
# travel with the userspace. Everything else ROCm needs is in the bundle, including
# libnuma, which the flatpak runtime and minimal distros do not ship.
host_packages="libdrm2 libdrm-amdgpu1 libelf1"

# binutils supplies the readelf the failure path uses to report runpaths.
diagnostic_packages="binutils"

# Run inside the container. Sent over stdin so the quoting stays readable.
loader_check() {
  cat <<'CHECK'
set -euo pipefail
apt-get update -qq
# shellcheck disable=SC2086
apt-get install -y -qq ${HOST_PACKAGES} ${DIAGNOSTIC_PACKAGES} >/dev/null

# Name what must be here rather than counting whatever the glob matched: two SONAME
# aliases of the backend would satisfy a count of two while llama-server was absent.
[ -e /bundle/llama-server ] || { echo "no llama-server in the bundle" >&2; exit 1; }
# ldd cannot see a dlopen: libamdhip64 loads comgr by SONAME at runtime, so its absence
# is invisible here and fatal on a card.
ls /bundle/libamd_comgr.so* >/dev/null 2>&1 || { echo "no libamd_comgr in the bundle" >&2; exit 1; }
ls /bundle/libggml-hip.so* >/dev/null 2>&1 || { echo "no libggml-hip.so in the bundle" >&2; exit 1; }

status=0
for lib in /bundle/libggml-hip.so* /bundle/llama-server; do
  [ -e "${lib}" ] || continue
  # ldd's own status matters: "not a dynamic executable" and a failed analysis both
  # print no "not found" line, so grepping alone treats them as success.
  rc=0
  resolved=$(ldd "${lib}" 2>&1) || rc=$?
  missing=$(echo "${resolved}" | grep "not found" || true)
  if [ -n "${missing}" ] || [ "${rc}" != 0 ]; then
    # Full output and runpaths, because "X => not found" for a file that IS in the
    # bundle means a search-path problem, and the next question is always whose.
    echo "UNRESOLVED in $(basename "${lib}") (ldd exit ${rc}):" >&2
    echo "${missing}" >&2
    echo "--- ldd ${lib}" >&2
    echo "${resolved}" >&2
    echo "--- runpaths in the bundle" >&2
    for obj in /bundle/*.so*; do
      echo "$(basename "${obj}"): $(readelf -d "${obj}" 2>/dev/null \
        | grep -E 'RUNPATH|RPATH' || echo 'none')" >&2
    done
    status=1
  fi
done

if [ "${status}" != 0 ]; then
  echo "the ROCm bundle does not close: the libraries above are neither beside the" >&2
  echo "binary nor installed by this script's host_packages" >&2
  exit 1
fi
echo "every ROCm dependency resolves with no ROCm installed"
CHECK
}

[ -d "${bundle_dir}" ] || { echo "no bundle directory at ${bundle_dir}" >&2; exit 1; }

loader_check | docker run --rm -i \
  -e HOST_PACKAGES="${host_packages}" \
  -e DIAGNOSTIC_PACKAGES="${diagnostic_packages}" \
  -v "$(cd "${bundle_dir}" && pwd):/bundle:ro" \
  "${image}" bash -s
