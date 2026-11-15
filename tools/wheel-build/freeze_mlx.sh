#!/usr/bin/env bash
# Build the mlx engine wheel on an Apple Silicon runner: a relocatable CPython with mlx-lm and its
# packages, in one wheel.
#
# Usage: freeze_mlx.sh <outdir>
#
# The PATH holds uv (the version named below), delocate-wheel (the version named below) and python3
# with the wheel package. The steps, in order:
#   1. Check that the pins agree: mlx-lm, and the mlx and mlx-metal files of mlx.requirements
#      against the macOS floor.
#   2. `uv python install` puts python-build-standalone in a scratch directory. The tree copies it.
#   3. `uv pip install --target` installs the hashed requirements into its site-packages.
#   4. collect_python_licences.py copies every licence text of the tree to THIRD-PARTY/mlx, names the
#      terms, and stops on a GPL-family licence.
#   5. `wheel pack` writes the wheel: the launcher bin/mlx-lm-server, python/, and the dist-info.
#   6. `delocate-wheel --require-target-macos-version` measures the floor of every Mach-O. Nothing
#      may be grafted.
#
# Output on stdout, last line: `WHEEL <path>`. <outdir> also receives terms.txt, the SPDX terms of
# the tree. The scratch directory is made under TMPDIR and left for the caller.
set -euo pipefail

readonly MLX_LM="0.32.2"
readonly MLX="0.32.4"
readonly UV="0.13.7"
readonly DELOCATE="0.13.0"
readonly PYTHON="3.12"
readonly FLOOR="14"
readonly ARCH="arm64"
readonly BACKEND="metal"
readonly OWN_TERMS="MIT"

die() {
  echo "freeze_mlx: $*" >&2
  exit 1
}

test "$#" -eq 1 || {
  echo "usage: freeze_mlx.sh <outdir>" >&2
  exit 2
}
outdir="$1"
here="$(cd "$(dirname "$0")" && pwd)"
requirements="${here}/mlx.requirements"
cp_tag="cp${PYTHON/./}"
platform="macosx_${FLOOR}_0_${ARCH}"

# The pins must agree before anything is installed.
grep -qxF "mlx-lm==${MLX_LM} \\" "${requirements}" ||
  die "mlx-lm ${MLX_LM} is not the pin of mlx.requirements. Make that file again with the new version."
grep -Eq "^mlx @ .*/mlx-${MLX}-${cp_tag}-${cp_tag}-${platform}\.whl \\\\\$" "${requirements}" ||
  die "mlx.requirements pins no mlx ${MLX} file mlx-${MLX}-${cp_tag}-${cp_tag}-${platform}.whl, the file of macOS ${FLOOR}."
grep -Eq "^mlx-metal @ .*/mlx_metal-${MLX}-py3-none-${platform}\.whl \\\\\$" "${requirements}" ||
  die "mlx.requirements pins no mlx-metal ${MLX} file mlx_metal-${MLX}-py3-none-${platform}.whl, the file of macOS ${FLOOR}."
case "$(uv --version)" in
  "uv ${UV}" | "uv ${UV} "*) ;;
  *) die "uv ${UV} is pinned in this script. The PATH has $(uv --version)." ;;
esac
delocate-wheel --version 2>&1 | grep -qF "${DELOCATE}" || die "delocate ${DELOCATE} is pinned in this script. The PATH has another version."

scratch="$(mktemp -d "${TMPDIR:-/tmp}/freeze.XXXXXX")"
root="${scratch}/pack/lilbee_engine_mlx-${MLX_LM}"
pkg="${root}/lilbee_engine_mlx"
dist="${root}/lilbee_engine_mlx-${MLX_LM}.dist-info"
site="${pkg}/python/lib/python${PYTHON}/site-packages"
mkdir -p "${pkg}/bin" "${dist}/licenses/THIRD-PARTY" "${outdir}"

UV_PYTHON_INSTALL_DIR="${scratch}/uv-python" uv python install "${PYTHON}"
interpreter="$(UV_PYTHON_INSTALL_DIR="${scratch}/uv-python" uv python find --managed-python "${PYTHON}")"
cp -R "$(cd "$(dirname "${interpreter}")/.." && pwd -P)" "${pkg}/python"
while IFS= read -r link; do
  ! test -d "${link}" || die "the interpreter tree holds the directory link ${link#"${pkg}"/}, which a wheel cannot keep"
done < <(find "${pkg}/python" -type l)
test -x "${pkg}/python/bin/python${PYTHON}" || die "the interpreter tree has no bin/python${PYTHON}"

uv pip install --python "${pkg}/python/bin/python${PYTHON}" --target "${site}" \
  --require-hashes --only-binary :all: --no-deps -r "${requirements}"
grep -q "^Tag: py3-none-${platform}\$" "${site}/mlx_metal-${MLX}.dist-info/WHEEL" ||
  die "mlx-metal ${MLX} was installed with another tag than ${platform}"

python3 -I "${here}/collect_python_licences.py" "${site}" "${dist}/licenses/THIRD-PARTY/mlx" "${outdir}/terms.txt" ||
  die "the tree holds a copyleft licence, a licence that the script cannot read, or a package with no licence file"
find "${pkg}/python" -maxdepth 4 -iname '*licen*' -not -path "${site}/*"
cp "${site}/mlx_lm-${MLX_LM}.dist-info/licenses/LICENSE" "${dist}/licenses/LICENSE-mlx-lm"
cp "${site}/mlx-${MLX}.dist-info/licenses/LICENSE" "${dist}/licenses/LICENSE-mlx"
cp "${pkg}/python/lib/python${PYTHON}/LICENSE.txt" "${dist}/licenses/LICENSE-Python"

# The expression names the terms of mlx-lm and mlx, then every other term of the tree.
expression="${OWN_TERMS}"
while IFS= read -r term; do
  [ "${term}" = "${OWN_TERMS}" ] || expression="${expression} AND ${term}"
done <"${outdir}/terms.txt"

# The launcher: a regular sh file that the store runs as <package>/bin/mlx-lm-server.
cat >"${pkg}/bin/mlx-lm-server" <<LAUNCHER
#!/bin/sh
here="\$(cd "\$(dirname "\$0")/.." && pwd)"
exec "\${here}/python/bin/python${PYTHON}" -I -m mlx_lm.server "\$@"
LAUNCHER
chmod 755 "${pkg}/bin/mlx-lm-server"
: >"${pkg}/__init__.py"
{
  echo "Metadata-Version: 2.4"
  echo "Name: lilbee-engine-mlx"
  echo "Version: ${MLX_LM}"
  echo "License-Expression: ${expression}"
  (cd "${dist}/licenses" && find . -type f | sed 's|^\./||' | sort | sed 's|^|License-File: |')
} >"${dist}/METADATA"
cat >"${dist}/WHEEL" <<WHL
Wheel-Version: 1.0
Generator: lilbee
Root-Is-Purelib: false
Tag: py3-none-${platform}
Build: 1.${BACKEND}
WHL

mkdir -p "${scratch}/packed" "${scratch}/repaired"
python3 -m wheel pack -d "${scratch}/packed" "${root}" >/dev/null
packed=("${scratch}"/packed/*.whl)
test "${#packed[@]}" -eq 1 || die "wheel pack wrote ${#packed[@]} wheels, expected one"

# delocate-wheel measures the floor of every Mach-O of the tree and stops when one needs more.
MACOSX_DEPLOYMENT_TARGET="${FLOOR}.0" delocate-wheel --require-archs "${ARCH}" \
  --require-target-macos-version "${FLOOR}.0" -w "${scratch}/repaired" "${packed[0]}" >"${scratch}/delocate.log" 2>&1 || {
  tail -5 "${scratch}/delocate.log" >&2
  die "delocate-wheel stopped. A Mach-O of the tree needs more than macOS ${FLOOR}.0, or a library is missing, or the tree lacks ${ARCH}."
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
