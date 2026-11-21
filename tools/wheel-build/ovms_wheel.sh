#!/usr/bin/env bash
# Put the OpenVINO Model Server release into the engine wheel project and build the wheel.
#
# Usage: ovms_wheel.sh <archive.tar.gz> <tbb-lib-dir> <platform-tag> <out-dir>
#
# Reads WHEEL_VERSION (for example 2026.4.1) and PYTHON (default python3), and the pin in
# engines/ovms.env (OVMS_ARCHIVE_SHA256, OVMS_BACKEND). <tbb-lib-dir> holds libtbb.so.12 from the
# PyPI tbb wheel: the archive does not ship it, and the server fails to start without it. The
# server binary finds its libraries through the run path $ORIGIN/../lib, so the wheel puts every
# library the binary needs, libtbb included, in lib/ beside bin/. Only the libraries that ldd
# resolves inside the archive are copied, each under the name the binary asks for.
# patchelf must be on PATH. Writes the unrepaired wheel in <out-dir>/dist and the licence expression in
# <out-dir>/license-expression.txt.
set -euo pipefail

test "$#" -eq 4 || {
  echo "usage: ovms_wheel.sh <archive.tar.gz> <tbb-lib-dir> <platform-tag> <out-dir>" >&2
  exit 2
}
archive="$1" tbb="$2" platform="$3" out="$4"
: "${WHEEL_VERSION:?}"
python="${PYTHON:-python3}"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "${here}/../.." && pwd)"
project="${root}/packaging/engine-wheel-ovms"
# shellcheck source=/dev/null
. "${root}/engines/ovms.env"
# bin/ovms links libgit2 (GPL-2.0 with a linking exception) and no build flag removes it. The
# wheel stays unbuilt until the owner rules on that licence.
test "${OVMS_LIBGIT2_RULING:-}" = "accepted" || {
  echo "ovms_wheel.sh: the wheel bundles libgit2 (GPL-2.0 with a linking exception)." >&2
  echo "Set OVMS_LIBGIT2_RULING=accepted after the owner accepts that licence." >&2
  exit 3
}
mkdir -p "${out}"
out="$(cd "${out}" && pwd)"

echo "${OVMS_ARCHIVE_SHA256}  ${archive}" | sha256sum --check --strict -
test -f "${tbb}/libtbb.so.12" || { echo "no libtbb.so.12 in ${tbb}" >&2; exit 1; }

unpack="$(mktemp -d)"
tar -xzf "${archive}" -C "${unpack}"
server="${unpack}/ovms"
test -x "${server}/bin/ovms" || { echo "the archive has no ovms/bin/ovms" >&2; exit 1; }

bin="${project}/lilbee_engine_ovms/bin" lib="${project}/lilbee_engine_ovms/lib"
mkdir -p "${bin}" "${lib}" "${project}/THIRD-PARTY"
cp "${server}/bin/ovms" "${bin}/ovms"
chmod 755 "${bin}/ovms"
cp "${tbb}/libtbb.so.12" "${lib}/libtbb.so.12"
# ldd resolves each name through the run path, so the copy follows the symlinks the archive uses.
LD_LIBRARY_PATH="${tbb}" ldd "${server}/bin/ovms" | while read -r name _ path _; do
  case "${path}" in
    "${server}"/lib/* | "${server}"/bin/../lib/*) cp -L "${path}" "${lib}/${name}" ;;
  esac
done
# The server loads the device plugins, the tokenizer library and the IR reader at run time, so ldd
# does not list them.
for path in "${server}"/lib/libopenvino_*plugin.so "${server}"/lib/libopenvino_tokenizers.so \
  "${server}"/lib/libopenvino_intel_npu_*.so "${server}"/lib/libopenvino_ir_frontend.so.*; do
  cp -L "${path}" "${lib}/$(basename "${path}")"
done
# Several libraries have no run path and need libtbb, so each library looks beside itself.
for path in "${lib}"/*; do
  chmod u+w "${path}"
  patchelf --set-rpath '$ORIGIN' "${path}"
done
# libespeak-ng is not copied: no library links it, and only the speech code of libopenvino_genai
# loads it by name. The licence step leaves out the texts of code that the wheel does not hold.
sh "${here}/ovms_licences.sh" "${server}" "${project}"
cp "${tbb}/../tbb-"*".dist-info/LICENSE.txt" "${project}/LICENSE-tbb" 2> /dev/null || true

expression="Apache-2.0 AND BSD-3-Clause AND GPL-2.0-only WITH GCC-exception-2.0"
echo "License-Expression: ${expression}"
printf '%s\n' "${expression}" > "${out}/license-expression.txt"

"${python}" "${here}/stamp_engine_wheel.py" "${project}/pyproject.toml" "${WHEEL_VERSION}" "${expression}"
(cd "${project}" && "${python}" -m build --wheel -C--build-option="--build-number=1.${OVMS_BACKEND//-/_}" \
  --outdir "${out}/dist")
"${python}" -m wheel tags --remove --platform-tag "${platform}" "${out}"/dist/*.whl
ls -l "${out}/dist"
