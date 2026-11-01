#!/usr/bin/env bash
# Pack the Visual C++ runtime the Windows engine imports into the wheel's bin/, so a
# fresh Windows machine needs a driver and nothing else. Fails, naming the DLL, when
# a redistributable file is absent. tools/qa/assert_engine_bundle.py checks the
# finished wheel for the same DLLs.
#
# Reads:
#   VCToolsRedistDir  the MSVC redistributable root (default: newest under Program Files)
# Argument:
#   the wheel's bin/ directory, holding the freshly built engine

set -euo pipefail

pkg_bin_dir="${1:?the wheel bin/ directory is required}"

if [ -n "${VCToolsRedistDir:-}" ]; then
  redist_dir="$(cygpath -u "${VCToolsRedistDir}")/x64"
else
  shopt -s nullglob
  redist_candidates=(/c/Program\ Files/Microsoft\ Visual\ Studio/*/*/VC/Redist/MSVC/*/x64)
  shopt -u nullglob
  redist_dir="${redist_candidates[${#redist_candidates[@]}-1]:-}"
fi
[ -d "${redist_dir}/Microsoft.VC143.CRT" ] || { echo "no MSVC CRT redistributable dir found (VCToolsRedistDir unset, no VC143 CRT under Program Files)" >&2; exit 1; }

# The CRT is newer than the copy a fresh Windows image ships in System32 (Server 2025
# has 14.22), and the engine faults inside the old DLL. vcomp140 is the OpenMP runtime
# ggml-base and ggml-cpu import; it lives in its own redist directory.
for dll in Microsoft.VC143.CRT/msvcp140.dll Microsoft.VC143.CRT/vcruntime140.dll \
  Microsoft.VC143.CRT/vcruntime140_1.dll Microsoft.VC143.OpenMP/vcomp140.dll; do
  [ -f "${redist_dir}/${dll}" ] || { echo "no ${dll##*/} under ${redist_dir}: the Windows bundle is incomplete" >&2; exit 1; }
  cp -L "${redist_dir}/${dll}" "${pkg_bin_dir}/"
done
