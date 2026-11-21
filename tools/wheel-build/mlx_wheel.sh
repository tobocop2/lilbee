#!/usr/bin/env bash
# Build the mlx engine wheel on Apple Silicon: a relocatable CPython with mlx-lm and its packages,
# in one wheel.
#
# Usage: mlx_wheel.sh <outdir>
#
# The pins are in engines/mlx.env; the package files and the hashed requirements are in
# packaging/engine-mlx/. The PATH holds uv and delocate-wheel at the versions of the pin, and
# python3 with the wheel package. The steps, in order:
#   1. Refuse a host that is not macOS on arm64, and pins that do not agree: the mlx-lm version,
#      and the mlx and mlx-metal files of the requirements against the macOS floor.
#   2. `uv python install` puts python-build-standalone in a scratch directory. The tree copies it.
#   3. `uv pip install --target` installs the hashed requirements into its site-packages.
#   4. collect_python_licences.py copies every licence text of the tree to THIRD-PARTY/mlx, names the
#      terms, and stops on a GPL-family licence.
#   5. `wheel pack` writes the wheel: the launcher bin/mlx-lm-server, python/, and the dist-info.
#   6. `delocate-wheel --require-target-macos-version` measures the floor of every Mach-O. Nothing
#      may be grafted.
#
# Output on stdout, last line: `WHEEL <path>`. <outdir> also receives terms.txt, the SPDX terms of
# the tree. The scratch directory is made under TMPDIR and left for the caller. Nothing is written
# outside the scratch directory and <outdir>.
set -euo pipefail

readonly OWN_TERMS="MIT"

die() {
  echo "mlx_wheel: $*" >&2
  exit 1
}

test "$#" -eq 1 || {
  echo "usage: mlx_wheel.sh <outdir>" >&2
  exit 2
}
outdir="$1"
here="$(cd "$(dirname "$0")" && pwd)"
top="$(cd "${here}/../.." && pwd)"
# shellcheck source=/dev/null
. "${top}/engines/mlx.env"
requirements="${top}/${MLX_REQUIREMENTS}"
project="${top}/packaging/engine-mlx"

host="$(uname -s)-$(uname -m)"
[ "${host}" = Darwin-arm64 ] || die "the mlx pack builds on macOS with Apple Silicon. This host is ${host}."
[[ "${MLX_FLOOR}" =~ ^([0-9]+)\.0$ ]] || die "MLX_FLOOR ${MLX_FLOOR} is not a whole macOS version such as 14.0"
platform="macosx_${BASH_REMATCH[1]}_0_arm64"
[ "${MLX_PLATFORM}" = "${platform}" ] ||
  die "MLX_PLATFORM ${MLX_PLATFORM} is not the tag of MLX_FLOOR ${MLX_FLOOR}, which is ${platform}."
cp_tag="cp${MLX_PYTHON/./}"

# The pins must agree before anything is installed.
grep -qxF "mlx-lm==${MLX_LM_VERSION} \\" "${requirements}" ||
  die "mlx-lm ${MLX_LM_VERSION} is not the pin of ${MLX_REQUIREMENTS}. Make that file again with the new version."
grep -Eq "^mlx @ .*/mlx-${MLX_VERSION}-${cp_tag}-${cp_tag}-${platform}\.whl \\\\\$" "${requirements}" ||
  die "${MLX_REQUIREMENTS} pins no mlx ${MLX_VERSION} file mlx-${MLX_VERSION}-${cp_tag}-${cp_tag}-${platform}.whl, the file of macOS ${MLX_FLOOR}."
grep -Eq "^mlx-metal @ .*/mlx_metal-${MLX_VERSION}-py3-none-${platform}\.whl \\\\\$" "${requirements}" ||
  die "${MLX_REQUIREMENTS} pins no mlx-metal ${MLX_VERSION} file mlx_metal-${MLX_VERSION}-py3-none-${platform}.whl, the file of macOS ${MLX_FLOOR}."
case "$(uv --version)" in
  "uv ${MLX_UV_VERSION}" | "uv ${MLX_UV_VERSION} "*) ;;
  *) die "uv ${MLX_UV_VERSION} is pinned in engines/mlx.env. The PATH has $(uv --version)." ;;
esac
delocate-wheel --version 2>&1 | grep -qF "${MLX_DELOCATE_VERSION}" ||
  die "delocate ${MLX_DELOCATE_VERSION} is pinned in engines/mlx.env. The PATH has another version."

scratch="$(mktemp -d "${TMPDIR:-/tmp}/mlx-wheel.XXXXXX")"
root="${scratch}/pack/lilbee_engine_mlx-${MLX_LM_VERSION}"
pkg="${root}/lilbee_engine_mlx"
dist="${root}/lilbee_engine_mlx-${MLX_LM_VERSION}.dist-info"
site="${pkg}/python/lib/python${MLX_PYTHON}/site-packages"
mkdir -p "${root}" "${dist}/licenses/THIRD-PARTY" "${outdir}"
cp -R "${project}/lilbee_engine_mlx" "${pkg}"
chmod 755 "${pkg}/bin/mlx-lm-server"

# uv also links each Python it installs into a bin folder, ~/.local/bin by default, so that
# folder moves into the scratch directory as well.
export UV_PYTHON_INSTALL_DIR="${scratch}/uv-python" UV_PYTHON_BIN_DIR="${scratch}/uv-bin"
uv python install "${MLX_PYTHON}"
interpreter="$(uv python find --managed-python "${MLX_PYTHON}")"
cp -R "$(cd "$(dirname "${interpreter}")/.." && pwd -P)" "${pkg}/python"
while IFS= read -r link; do
  ! test -d "${link}" || die "the interpreter tree holds the directory link ${link#"${pkg}"/}, which a wheel cannot keep"
done < <(find "${pkg}/python" -type l)
test -x "${pkg}/python/bin/python3" || die "the interpreter tree has no bin/python3, which the launcher runs"

uv pip install --python "${pkg}/python/bin/python${MLX_PYTHON}" --target "${site}" \
  --require-hashes --only-binary :all: --no-deps -r "${requirements}"
grep -q "^Tag: py3-none-${platform}\$" "${site}/mlx_metal-${MLX_VERSION}.dist-info/WHEEL" ||
  die "mlx-metal ${MLX_VERSION} was installed with another tag than ${platform}"

# supplied-licences.txt replaces the numpy licence because this wheel bundles no library of its own.
! test -e "${site}/numpy/.dylibs" ||
  die "numpy bundles libraries in numpy/.dylibs, so its own LICENSE.txt must stay. Change packaging/engine-mlx/supplied-licences.txt."
# The bundled interpreter runs the collector, so the Python on PATH may be older than 3.10.
"${pkg}/python/bin/python3" -I "${here}/collect_python_licences.py" "${site}" "${dist}/licenses/THIRD-PARTY/mlx" \
  "${outdir}/terms.txt" "${project}/supplied-licences.txt" ||
  die "the tree holds a copyleft licence, a licence that the script cannot read, or a package with no licence file"
cp "${site}/mlx_lm-${MLX_LM_VERSION}.dist-info/licenses/LICENSE" "${dist}/licenses/LICENSE-mlx-lm"
cp "${site}/mlx-${MLX_VERSION}.dist-info/licenses/LICENSE" "${dist}/licenses/LICENSE-mlx"
cp "${pkg}/python/lib/python${MLX_PYTHON}/LICENSE.txt" "${dist}/licenses/LICENSE-Python"
# The engine store installs an MPL-2.0 tree only with this file: one PyPI source archive URL and its
# sha256 for each MPL-2.0 package (tqdm), as engine_crates.MPL_PYPI_CHECKSUMS lists them.
if grep -qx "MPL-2.0" "${outdir}/terms.txt"; then
  cp "${project}/SOURCE-MPL-2.0" "${dist}/licenses/SOURCE-MPL-2.0"
fi

# The expression names the terms of mlx-lm and mlx, then every other term of the tree.
expression="${OWN_TERMS}"
while IFS= read -r term; do
  [ "${term}" = "${OWN_TERMS}" ] || expression="${expression} AND ${term}"
done <"${outdir}/terms.txt"

{
  echo "Metadata-Version: 2.4"
  echo "Name: lilbee-engine-mlx"
  echo "Version: ${MLX_LM_VERSION}"
  echo "License-Expression: ${expression}"
  (cd "${dist}/licenses" && find . -type f | sed 's|^\./||' | sort | sed 's|^|License-File: |')
} >"${dist}/METADATA"
{
  echo "Wheel-Version: 1.0"
  echo "Generator: lilbee"
  echo "Root-Is-Purelib: false"
  echo "Tag: py3-none-${platform}"
  echo "Build: 1.${MLX_BACKEND}"
} >"${dist}/WHEEL"

mkdir -p "${scratch}/packed" "${scratch}/repaired"
python3 -m wheel pack -d "${scratch}/packed" "${root}" >/dev/null
packed=("${scratch}"/packed/*.whl)
test "${#packed[@]}" -eq 1 || die "wheel pack wrote ${#packed[@]} wheels, expected one"

# delocate-wheel measures the floor of every Mach-O of the tree and stops when one needs more.
MACOSX_DEPLOYMENT_TARGET="${MLX_FLOOR}" delocate-wheel --require-archs arm64 \
  --require-target-macos-version "${MLX_FLOOR}" -w "${scratch}/repaired" "${packed[0]}" >"${scratch}/delocate.log" 2>&1 || {
  tail -5 "${scratch}/delocate.log" >&2
  die "delocate-wheel stopped. A Mach-O of the tree needs more than macOS ${MLX_FLOOR}, or a library is missing, or the tree lacks arm64."
}
repaired=("${scratch}"/repaired/*.whl)
test "${#repaired[@]}" -eq 1 || die "delocate-wheel wrote ${#repaired[@]} wheels, expected one"
final="${repaired[0]}"
case "${final}" in
  *-"${platform}.whl") ;;
  *) die "the repaired wheel $(basename "${final}") does not carry ${platform}" ;;
esac
python3 -m zipfile -l "${final}" | awk '{print $1}' >"${scratch}/final-listing.txt"
if grep -q '\.dylibs/.' "${scratch}/final-listing.txt"; then
  die "delocate-wheel grafted a library into the wheel. A library of the tree must come from the tree or the system."
fi
mv "${final}" "${outdir}/"
echo "WHEEL ${outdir}/$(basename "${final}")"
