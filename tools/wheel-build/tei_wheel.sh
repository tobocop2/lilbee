#!/usr/bin/env bash
# Put a built text-embeddings-router and its licences into the engine wheel project and build the wheel.
#
# Usage: tei_wheel.sh <tei-source-dir> <router-binary> <cargo-features> <target-triple> <backend> <platform-tag> <out-dir>
#
# Reads WHEEL_VERSION (for example 1.9.4) and PYTHON (default python3). NOTICE-NVIDIA-CUDA must
# already sit in the project when the build holds NVIDIA files; the wheel then names the NVIDIA
# licence. OPENSSL_LICENSE names a licence file to ship as LICENSE-openssl (a Linux wheel that
# grafts the system OpenSSL). The licence expression is Apache-2.0 (the router) and every licence
# that cargo-about found. Writes <out-dir>/license-expression.txt and the unrepaired wheel.
# cargo-about, jq and cargo must be on PATH, and the python build module must be installed.
set -euo pipefail

test "$#" -eq 7 || {
  echo "usage: tei_wheel.sh <tei-source-dir> <router-binary> <cargo-features> <target-triple> <backend> <platform-tag> <out-dir>" >&2
  exit 2
}
src="$1" binary="$2" features="$3" target="$4" backend="$5" platform="$6" out="$7"
: "${WHEEL_VERSION:?}"
python="${PYTHON:-python3}"
here="$(cd "$(dirname "$0")" && pwd)"
project="$(cd "${here}/../../packaging/engine-wheel-tei" && pwd)"
mkdir -p "${out}" "${project}/lilbee_engine_tei/bin"
# The wheel build below starts in the project directory, so <out-dir> must be absolute.
out="$(cd "${out}" && pwd)"

cp "${src}/LICENSE" "${project}/LICENSE-tei"
if [ -n "${OPENSSL_LICENSE:-}" ]; then
  cp "${OPENSSL_LICENSE}" "${project}/LICENSE-openssl"
fi
bash "${here}/rust_licences.sh" "${src}" "${features}" "${target}" "${project}/THIRD-PARTY/tei" "${out}/terms.txt"
"${python}" "${here}/mpl_notice.py" "${out}/terms.json" "${src}/Cargo.lock" "${project}/SOURCE-MPL-2.0"
bash "${here}/vendored_notices.sh" "${src}" "${project}/THIRD-PARTY/tei-notices"

expression="Apache-2.0"
while IFS= read -r term; do
  case " ${expression//AND/} " in *" ${term} "*) continue ;; esac
  expression="${expression} AND ${term}"
done < "${out}/terms.txt"
if [ -f "${project}/NOTICE-NVIDIA-CUDA" ]; then
  expression="${expression} AND LicenseRef-NVIDIA-CUDA-EULA"
fi
echo "License-Expression: ${expression}"
printf '%s\n' "${expression}" > "${out}/license-expression.txt"

cp "${binary}" "${project}/lilbee_engine_tei/bin/text-embeddings-router"
chmod 755 "${project}/lilbee_engine_tei/bin/text-embeddings-router"
"${python}" "${here}/stamp_engine_wheel.py" "${project}/pyproject.toml" "${WHEEL_VERSION}" "${expression}"
# The build and the retag stamp every wheel entry with the commit time of the source.
export "$(bash "${here}/source_date_epoch.sh" "${src}")"
(cd "${project}" && "${python}" -m build --wheel -C--build-option="--build-number=1.${backend//-/_}" \
  --outdir "${out}/dist")
"${python}" -m wheel tags --remove --platform-tag "${platform}" "${out}"/dist/*.whl
ls -l "${out}/dist"
