#!/usr/bin/env bash
# Build and check the mistral.rs engine pack of one cell. Run it from a fresh lilbee checkout:
#
#   OUT=/path/to/out BACKEND=cpu ... bash tools/wheel-build/pack_mistralrs.sh
#
# The engine-packs repository sets the variables from its cells file (scripts/pack.sh):
#   MISTRALRS_REPO, MISTRALRS_TAG   the upstream release
#   ASSET                           the archive of that release for this cell
#   BACKEND, PLATFORM               the backend and the platform tag of the wheel
#   RUST_TARGET, RUST_FEATURES      the crate graph of the binary (RUST_FEATURES may be empty)
#   CARGO_ABOUT_VERSION             the cargo-about release
#   ABOUT_TARGET                    the cargo-about build for this host
#   MISTRALRS_CHAT_MODEL            the model of the chat check
#   OUT                             the folder that receives the checked wheel and its sha256 file
#
# The script needs Python 3.12, cargo, git and a gh login. It stamps the wheel project in place, so it
# refuses a checkout that an earlier build used.

set -euo pipefail

# The mode of each file goes into the wheel.
umask 022

# shellcheck source=tools/wheel-build/pack_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/pack_common.sh"

WHEEL_TREE=packaging/engine-wheel-mistral-rs

require_inputs() {
  local name
  host_os > /dev/null
  for name in MISTRALRS_REPO MISTRALRS_TAG ASSET BACKEND PLATFORM RUST_TARGET CARGO_ABOUT_VERSION ABOUT_TARGET \
    MISTRALRS_CHAT_MODEL OUT; do
    [ -n "${!name:-}" ] || { echo "pack_mistralrs: ${name} is not set" >&2; return 2; }
  done
  for name in upstream cargo-about mistral.rs pack-venv out; do
    [ ! -e "${name}" ] || { echo "pack_mistralrs: an earlier build used this checkout (${name} exists). Use a fresh one." >&2; return 2; }
  done
}

fetch_cargo_about() {
  local name tool
  name="cargo-about-${CARGO_ABOUT_VERSION}-${ABOUT_TARGET}.tar.gz"
  fetch_release_asset EmbarkStudios/cargo-about "${CARGO_ABOUT_VERSION}" "${name}" cargo-about
  tar -xzf "cargo-about/${name}" -C cargo-about
  tool="$(find cargo-about -type f \( -name cargo-about -o -name cargo-about.exe \) | head -n 1)"
  [ -n "${tool}" ] || { echo "pack_mistralrs: ${name} holds no cargo-about" >&2; return 1; }
  CARGO_ABOUT_BIN="${PWD}/${tool}"
}

collect_licences() {
  local third="${WHEEL_TREE}/THIRD-PARTY/mistral.rs" features=()
  git clone -q --depth 1 --branch "${MISTRALRS_TAG}" "https://github.com/${MISTRALRS_REPO}.git" mistral.rs
  if [ -n "${RUST_FEATURES:-}" ]; then features=(--features "${RUST_FEATURES}"); fi
  # The graph is the one that the release job builds: the locked crates for this target and these features.
  (cd mistral.rs && "${CARGO_ABOUT_BIN}" generate --format json --locked \
    -c ../tools/wheel-build/mistralrs-about.toml \
    -m mistralrs-cli/Cargo.toml --target "${RUST_TARGET}" ${features[@]+"${features[@]}"} \
    -o ../about.json)
  python tools/wheel-build/rust_licences.py about.json "${third}" | tr -d '\r' > crate-licences.txt
  cat crate-licences.txt
  cp mistral.rs/LICENSE "${WHEEL_TREE}/LICENSE-mistral.rs"
  python tools/wheel-build/rust_notices.py about.json mistral.rs/Cargo.lock mistral.rs "${WHEEL_TREE}" | tr -d '\r' > notices.txt
  cat notices.txt
  find "${third}" -type f | wc -l
}

# Copy the tool into the wheel tree and export the time of the tool as SOURCE_DATE_EPOCH.
repack() {
  local line
  line="$(python tools/wheel-build/repack_mistralrs.py "upstream/${ASSET}" "${WHEEL_TREE}/lilbee_engine_mistral_rs/bin" | tr -d '\r')"
  [[ "${line}" =~ ^SOURCE_DATE_EPOCH=[0-9]+$ ]] || { echo "pack_mistralrs: no time of the tool in '${line}'" >&2; return 1; }
  export SOURCE_DATE_EPOCH="${line#SOURCE_DATE_EPOCH=}"
  ls -lR "${WHEEL_TREE}/lilbee_engine_mistral_rs/bin"
}

add_windows_runtime() {
  local bin="${WHEEL_TREE}/lilbee_engine_mistral_rs/bin"
  [ "$(host_os)" = Windows ] || return 0
  bash tools/wheel-build/bundle_windows_runtime.sh "${bin}" "${WHEEL_TREE}/NOTICE-MICROSOFT-VC-RUNTIME"
  # The OpenMP runtime stays only when the exe imports it.
  if ! grep -qiaF vcomp140 "${bin}/mistralrs.exe"; then rm -f "${WHEEL_TREE}/lilbee_engine_mistral_rs/bin/vcomp140.dll"; fi
  ls "${bin}"
}

# Set LICENSE_EXPRESSION and refuse an empty licence file.
name_licences() {
  local extra=(Apache-2.0 BSD-3-Clause) file
  if [ "$(host_os)" = Windows ]; then extra+=(LicenseRef-Microsoft-VC-Runtime); fi
  LICENSE_EXPRESSION="$({ echo MIT; cat crate-licences.txt; printf '%s\n' "${extra[@]}"; } | merge_terms)"
  export LICENSE_EXPRESSION
  echo "License-Expression: ${LICENSE_EXPRESSION}"
  for file in "${WHEEL_TREE}"/LICENSE-* "${WHEEL_TREE}"/NOTICE-* "${WHEEL_TREE}"/SOURCE-*; do
    [ -e "${file}" ] || continue
    [ -s "${file}" ] || { echo "empty licence file ${file}" >&2; return 1; }
  done
}

# Set the version, the licence expression and the licence files in the wheel project.
stamp_project() {
  (cd "${WHEEL_TREE}" && python - "${MISTRALRS_TAG#v}" <<'PY'
import os, pathlib, re, sys
p = pathlib.Path("pyproject.toml")
text = re.sub(r'(?m)^version = ".*"$', f'version = "{sys.argv[1]}"', p.read_text(), count=1)
files = '["LICENSE-*", "NOTICE-*", "SOURCE-*", "THIRD-PARTY/**"]'
text = text.replace('requires-python', 'license = "%s"\nlicense-files = %s\nrequires-python' % (os.environ["LICENSE_EXPRESSION"], files), 1)
p.write_text(text)
PY
  )
}

build_wheel() {
  (
    cd "${WHEEL_TREE}"
    # A relative path: Windows Python does not read a Git Bash path.
    install_build_tools python ../../tools/wheel-build/build-tools.constraints
    build_wheel_with_tools python -C--build-option="--build-number=1.${BACKEND}"
    python -m wheel tags --remove --platform-tag "${PLATFORM}" dist/*.whl
    ls dist
  )
}

main() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.."
  require_inputs
  use_build_python
  fetch_release_asset "${MISTRALRS_REPO}" "${MISTRALRS_TAG}" "${ASSET}" upstream
  ls -l upstream
  fetch_cargo_about
  collect_licences
  repack
  add_windows_runtime
  name_licences
  stamp_project
  build_wheel
  collect_wheel "${WHEEL_TREE}/dist" out
  bash tools/wheel-build/check_mistralrs_wheel.sh out/*.whl crate-licences.txt "${LICENSE_EXPRESSION}"
  # Only a wheel that passed its check reaches the output folder.
  mkdir -p "${OUT}"
  cp out/* "${OUT}/"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main
fi
