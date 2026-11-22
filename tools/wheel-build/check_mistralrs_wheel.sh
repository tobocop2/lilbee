#!/usr/bin/env bash
# Check one mistral.rs engine wheel: its tag, its licences, and that the tool starts on this computer.
#
#   check_mistralrs_wheel.sh <wheel> <crate-licences> <licence-expression>
#
# <crate-licences> is the list of licence ids that rust_licences.py prints, one per line.
# Reads:
#   BACKEND, PLATFORM      the backend and the platform tag that the wheel must have
#   MISTRALRS_CHAT_MODEL   the model of the chat check, which a cpu wheel must answer

set -euo pipefail

# shellcheck source=tools/wheel-build/pack_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/pack_common.sh"

refuse() {
  echo "check_mistralrs_wheel: $1" >&2
  return 1
}

check_tag() {  # wheel
  local name info
  name="$(basename "$1")"
  case "${name}" in
    lilbee_engine_mistral_rs-*-1."${BACKEND}"-py3-none-"${PLATFORM}".whl) ;;
    *) refuse "wrong tag on ${name}: expected backend ${BACKEND} and platform ${PLATFORM}" || return 1 ;;
  esac
  info="$(dist_info "$1" WHEEL)" || return 1
  grep -qxF "Tag: py3-none-${PLATFORM}" <<< "${info}" || refuse "the WHEEL file of ${name} lacks the tag py3-none-${PLATFORM}"
}

check_licences() {  # wheel crate-licences expression
  local wheel="$1" crates="$2" expression="$3" listed listing metadata pair
  if ! listed="$(cat "${crates}")" || [ -z "${listed}" ]; then
    refuse "no crate licences in ${crates}"
    return 1
  fi
  listing="$(python -m zipfile -l "${wheel}")" || return 1
  metadata="$(dist_info "${wheel}" METADATA)" || return 1
  grep -xF "License-Expression: ${expression}" <<< "${metadata}" || refuse "wheel lacks the licence expression ${expression}" || return 1
  grep -q "dist-info/licenses/LICENSE-mistral.rs " <<< "${listing}" || refuse "wheel lacks the mistral.rs licence" || return 1
  grep -q "dist-info/licenses/THIRD-PARTY/mistral.rs/" <<< "${listing}" || refuse "wheel lacks the crate licences" || return 1
  grep -q "dist-info/licenses/THIRD-PARTY/vendored/" <<< "${listing}" || refuse "wheel lacks the vendored notices" || return 1
  for pair in MPL-2.0:SOURCE-MPL-2.0 OpenSSL:NOTICE-OpenSSL; do
    if grep -qx "${pair%%:*}" <<< "${listed}"; then
      grep -q "dist-info/licenses/${pair#*:} " <<< "${listing}" || refuse "wheel lacks ${pair#*:}" || return 1
    fi
  done
  if [ "$(host_os)" = Windows ]; then
    grep -q "dist-info/licenses/NOTICE-MICROSOFT-VC-RUNTIME " <<< "${listing}" || refuse "wheel lacks the Visual C++ runtime notice"
  fi
}

# Fail when a library of the output of `otool -L` is outside the system folders.
only_system_libraries() {
  ! awk 'NR > 1 {print $1}' | grep -v -E '^(/usr/lib/|/System/Library/)'
}

start_tool() {  # wheel
  local os tools work py tool exe bindir dll libraries
  os="$(host_os)"
  tools="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  work="$(mktemp -d "${PWD}/pack-check.XXXXXX")"
  python -m venv "${work}/venv"
  if [ "${os}" = Windows ]; then py="${work}/venv/Scripts/python.exe"; tool=mistralrs.exe; else py="${work}/venv/bin/python"; tool=mistralrs; fi
  "${py}" -m pip install -q "$1"
  exe="$("${py}" -c "import lilbee_engine_mistral_rs as m, pathlib; print((pathlib.Path(m.__file__).parent / 'bin' / '${tool}').as_posix())" | tr -d '\r')"
  echo "mistralrs: ${exe}"
  [ -f "${exe}" ] || refuse "the installed wheel lacks ${tool}"
  if [ "${os}" != Windows ]; then [ -x "${exe}" ] || refuse "the installed tool is not executable"; fi
  bindir="$(dirname "${exe}")"
  ls -R "${bindir}"
  if [ "${os}" = Windows ]; then
    for dll in msvcp140.dll vcruntime140.dll vcruntime140_1.dll; do
      [ -f "${bindir}/${dll}" ] || refuse "wheel lacks ${dll}"
    done
  fi
  if [ "${os}" = macOS ]; then
    libraries="$(otool -L "${exe}")"
    echo "${libraries}"
    only_system_libraries <<< "${libraries}" || refuse "the binary loads a library outside the system"
  fi
  cd /
  "${exe}" --version || refuse "the installed tool does not start"
  # A Metal cell needs a device to serve a model: it stops at the version and the library check.
  if [ "${BACKEND}" = cpu ]; then
    HF_HOME="${RUNNER_TEMP:-${work}}/hf" python "${tools}/qa/check_mistralrs_chat.py" "${exe}" "${MISTRALRS_CHAT_MODEL:?}"
  fi
}

main() {
  if [ "$#" -ne 3 ]; then
    echo "usage: check_mistralrs_wheel.sh <wheel> <crate-licences> <licence-expression>" >&2
    return 2
  fi
  check_tag "$1"
  check_licences "$@"
  start_tool "$1"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
