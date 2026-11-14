#!/usr/bin/env bash
# Pack the Visual C++ runtime the Windows engine imports into the wheel's bin/, so a
# fresh Windows machine needs a driver and nothing else. Fails, naming the DLL, when
# a redistributable file is absent. tools/qa/assert_engine_bundle.py checks the
# finished wheel for the same DLLs.
#
# Reads:
#   VCToolsRedistDir  the MSVC redistributable root (default: the newest under the
#                     Visual Studio install that vswhere reports)
#   VSWHERE           the vswhere.exe path (default: the Visual Studio Installer copy)
# Argument:
#   the wheel's bin/ directory, holding the freshly built engine

set -euo pipefail

pkg_bin_dir="${1:?the wheel bin/ directory is required}"

redist_dir=""
vs_path=""
if [ -n "${VCToolsRedistDir:-}" ]; then
  redist_dir="$(cygpath -u "${VCToolsRedistDir}")/x64"
else
  # A CI step runs outside a developer shell, so ask vswhere for the install.
  vswhere="${VSWHERE:-/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe}"
  if [ -x "${vswhere}" ]; then
    vs_path="$("${vswhere}" -latest -products '*' \
      -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
      -property installationPath | tr -d '\r')"
  fi
  [ -n "${vs_path}" ] || { echo "no Visual Studio with the C++ tools found (VCToolsRedistDir unset, vswhere reported no install)" >&2; exit 1; }
  vs_path="$(cygpath -u "${vs_path}")"
  # Version folders only: this skips v143 and the onecore and debug_nonredist subfolders.
  redist_dir="$(find "${vs_path}/VC/Redist/MSVC" -mindepth 2 -maxdepth 2 -type d \
    -path '*/MSVC/[0-9]*/x64' 2>/dev/null | sort -V | tail -n 1 || true)"
  [ -n "${redist_dir}" ] || { echo "no MSVC redistributable folder under ${vs_path}" >&2; exit 1; }
fi

# The toolset number in the folder name changes with each Visual Studio (VC143, VC145), so match it.
find_redist_subdir() {
  find "${redist_dir}" -mindepth 1 -maxdepth 1 -type d -name "Microsoft.VC*.$1" 2>/dev/null | sort -V | tail -n 1 || true
}
crt_dir="$(find_redist_subdir CRT)"
omp_dir="$(find_redist_subdir OpenMP)"
[ -n "${crt_dir}" ] || { echo "no Microsoft.VC*.CRT folder under ${redist_dir}" >&2; exit 1; }
[ -n "${omp_dir}" ] || { echo "no Microsoft.VC*.OpenMP folder under ${redist_dir}" >&2; exit 1; }

# The CRT is newer than the copy a fresh Windows image ships in System32 (Server 2025
# has 14.22), and the engine faults inside the old DLL. vcomp140 is the OpenMP runtime
# ggml-base and ggml-cpu import; it lives in its own redist directory.
for dll in "${crt_dir}/msvcp140.dll" "${crt_dir}/vcruntime140.dll" \
  "${crt_dir}/vcruntime140_1.dll" "${omp_dir}/vcomp140.dll"; do
  [ -f "${dll}" ] || { echo "no ${dll##*/} under ${dll%/*}: the Windows bundle is incomplete" >&2; exit 1; }
  cp -L "${dll}" "${pkg_bin_dir}/"
done
