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
#   VS_LICENSE_URL    the Visual Studio Community licence .docx for the notice (default: the 2022 file)
#   VS_LICENSE_SHA256 the digest that file must have (default: the digest of the 2022 file)
#   PYTHON            the interpreter that reads the licence (default: python)
# Argument:
#   the wheel's bin/ directory, holding the freshly built engine
#   optional second: the path that receives the Microsoft redistribution notice
#   optional third: the path that receives a copy of the verified licence file, for
#                   vs_distributable_terms.py --verify on the finished wheel

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

# The redistribution terms are in the REDIST.TXT of the Visual Studio install that supplied the files.
notice_file="${2:-}"
if [ -n "${notice_file}" ]; then
  vs_root="$(cd "${redist_dir}/../../../../.." && pwd)"
  redist_txt="$(find "${vs_root}" -maxdepth 5 -iname 'redist*.txt' 2>/dev/null | sort | head -n 1 || true)"
  [ -n "${redist_txt}" ] || {
    echo "no REDIST.TXT under ${vs_root}; licence-like files:" >&2
    find "${vs_root}" -maxdepth 5 -iname '*licen*' 2>/dev/null | head -n 20 >&2
    exit 1
  }
  [ -s "${redist_txt}" ] || { echo "${redist_txt} is empty" >&2; exit 1; }
  # The Distributable Code terms come from the Visual Studio Community licence, pinned by digest.
  work="$(mktemp -d)"
  license_url="${VS_LICENSE_URL:-https://visualstudio.microsoft.com/wp-content/uploads/2021/11/Visual-Studio-2022-Community-License-EN.docx}"
  license_sha256="${VS_LICENSE_SHA256:-41a207b10c8ab91d0d2f10a854715f73dca54509581692d2fe179aa3ffcb8540}"
  curl -fsSL --retry 3 --retry-delay 5 "${license_url}" -o "${work}/license.docx"
  if command -v sha256sum >/dev/null; then
    got="$(sha256sum "${work}/license.docx" | cut -d' ' -f1)"
  else
    got="$(shasum -a 256 "${work}/license.docx" | cut -d' ' -f1)"
  fi
  [ "${got}" = "${license_sha256}" ] || { echo "the Visual Studio licence has digest ${got}, not ${license_sha256}" >&2; exit 1; }
  # The tool writes the terms as UTF-8 itself: a redirected stdout uses the runner's locale encoding.
  "${PYTHON:-python}" -I "$(dirname "${BASH_SOURCE[0]}")/vs_distributable_terms.py" "${work}/license.docx" "${work}/terms.txt"
  [ -z "${3:-}" ] || cp "${work}/license.docx" "$3"
  {
    echo "These files are licensed under the Microsoft terms below, not under the other terms of this wheel."
    echo
    echo "This wheel carries Microsoft Visual C++ runtime files: msvcp140.dll, vcruntime140.dll and"
    echo "vcruntime140_1.dll, and vcomp140.dll when the wheel holds it. They are distributable code"
    echo "under the Microsoft Visual Studio Community 2022 licence terms ${license_url}"
    echo "The files are not modified."
    echo
    echo "Distributable Code section of those terms:"
    echo
    cat "${work}/terms.txt"
    echo
    echo "The list that follows is ${redist_txt##*/} from the Visual Studio install that supplied the files."
    echo
    tr -d '\r' < "${redist_txt}"
  } > "${notice_file}"
  echo "Visual C++ runtime notice: ${redist_txt}, $(wc -c < "${redist_txt}") bytes"
fi
