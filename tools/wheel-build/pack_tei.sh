#!/usr/bin/env bash
# Build and check the text-embeddings-inference engine pack of one cell. Run it from a fresh lilbee checkout:
#
#   OUT=/path/to/out BACKEND=cpu ... bash tools/wheel-build/pack_tei.sh
#
# The engine-packs repository sets the variables from its cells file (scripts/pack.sh):
#   TEI_REPO, TEI_TAG          the router source
#   RUSTC_VERSION              the Rust toolchain, which rust-toolchain.toml of the tag must name
#   WHEEL_VERSION              the version of the wheel
#   BACKEND, PLATFORM          the backend and the platform tag of the wheel
#   CUDA_CC                    the CUDA compute capability of a Linux cell; empty for the CPU cell
#   CUDA_VERSION, CUDA_PACKAGE_SUFFIX   the CUDA toolkit of the Linux cells
#   MANYLINUX_IMAGE            the build image of the Linux cells
#   MACOSX_DEPLOYMENT_TARGET   the macOS floor of the macOS cell
#   OUT                        the folder that receives the checked wheel, its sha256 file and its licence expression
#
# The script needs Python 3.12 and git. A Linux cell also needs docker and qemu-x86_64-static, the macOS cell
# Xcode, rustup and jq. The macOS cell builds under ${TMPDIR}/lilbee-tei-<tag>, because Rust writes the source paths
# into the router. The build writes into the wheel project, so the script refuses a checkout that an earlier build used.

set -euo pipefail

# The mode of each file goes into the wheel.
umask 022

# shellcheck source=tools/wheel-build/pack_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/pack_common.sh"

WHEEL_TREE=packaging/engine-wheel-tei

# The folder of the macOS build.
metal_work() {
  local base="${TMPDIR:-/tmp}"
  printf '%s\n' "${base%/}/lilbee-tei-${TEI_TAG}"
}

require_inputs() {
  local os name names tools used=(out wheel pack-venv check-venv)
  os="$(host_os)"
  names=(TEI_REPO TEI_TAG RUSTC_VERSION WHEEL_VERSION BACKEND PLATFORM OUT)
  case "${os}" in
    Linux)
      names+=(CUDA_VERSION CUDA_PACKAGE_SUFFIX MANYLINUX_IMAGE)
      tools=(git docker qemu-x86_64-static)
      ;;
    macOS)
      names+=(MACOSX_DEPLOYMENT_TARGET)
      tools=(git rustup jq xcodebuild)
      ;;
    *)
      echo "pack_tei: no cell builds on ${os}" >&2
      return 2
      ;;
  esac
  for name in "${names[@]}"; do
    [ -n "${!name:-}" ] || { echo "pack_tei: ${name} is not set" >&2; return 2; }
  done
  for name in "${tools[@]}"; do
    command -v "${name}" > /dev/null || { echo "pack_tei: ${name} is not installed" >&2; return 2; }
  done
  if [ "${os}" = macOS ]; then used+=("$(metal_work)"); fi
  for name in "${used[@]}"; do
    [ ! -e "${name}" ] || { echo "pack_tei: an earlier build used this checkout or folder (${name} exists). Use a fresh checkout, and move the folder away." >&2; return 2; }
  done
}

# Build the repaired wheel in the manylinux image. tei_manylinux.sh installs the pinned build tools there.
build_linux() {
  docker run --rm -v "${PWD}:/work" -w /work \
    -e TEI_REPO -e TEI_TAG -e RUSTC_VERSION -e WHEEL_VERSION -e BACKEND -e CUDA_CC -e CUDA_PACKAGE_SUFFIX -e CUDA_VERSION \
    -e WORK=/work -e OUT=/work/out -e "OWNER=$(id -u):$(id -g)" \
    "${MANYLINUX_IMAGE}" \
    bash /work/tools/wheel-build/tei_manylinux.sh
  ls -l out out/repaired
  df -h / | tail -1
}

# Build the repaired wheel on this Mac. The subshell keeps the build variables away from the checks.
build_metal() (
  work="$(metal_work)"
  src="${work}/src"
  features=candle,metal,http
  mkdir -p "${work}/bin"
  xcodebuild -version
  rustup toolchain install "${RUSTC_VERSION}" --profile minimal
  # A variable of the computer must not change the build. CARGO_HOME starts empty: vendored_notices.sh copies
  # the notices of every crate in it, and a cargo config in it changes the flags.
  unset "${!CARGO_@}" RUSTC RUSTC_WRAPPER RUSTC_WORKSPACE_WRAPPER
  export CARGO_HOME="${work}/cargo" RUSTUP_TOOLCHAIN="${RUSTC_VERSION}" LC_ALL=C
  bash tools/wheel-build/install_cargo_about.sh "${work}/bin"
  export PATH="${work}/bin:${PATH}"
  git clone -q --depth 1 --branch "${TEI_TAG}" "${TEI_REPO}" "${src}"
  grep -q "channel = \"${RUSTC_VERSION}\"" "${src}/rust-toolchain.toml" \
    || { echo "pack_tei: rust-toolchain.toml of ${TEI_TAG} does not name Rust ${RUSTC_VERSION}" >&2; exit 1; }
  # The upstream .cargo/config.toml builds for the CPU of the build host (target-cpu=native); RUSTFLAGS replaces it.
  export RUSTFLAGS="-C target-cpu=apple-m1"
  (cd "${src}" && cargo build --release --locked --bin text-embeddings-router --no-default-features --features "${features}")
  # A cargo that is not the rustup proxy ignores RUSTUP_TOOLCHAIN. Rust writes the commit of its compiler into the router.
  pinned="/rustc/$(rustup run "${RUSTC_VERSION}" rustc -vV | sed -n 's/^commit-hash: //p')/"
  compiled="$({ grep -aoE '/rustc/[0-9a-f]{40}/' "${src}/target/release/text-embeddings-router" || true; } | sort -u)"
  [ "${compiled}" = "${pinned}" ] \
    || { echo "pack_tei: the router comes from ${compiled:-no known compiler}, not from Rust ${RUSTC_VERSION} (${pinned})" >&2; exit 1; }
  install_build_tools python tools/wheel-build/build-tools.constraints tools/wheel-build/tei-tools-macos.constraints
  # The wheel build, the retag and delocate stamp every entry with the commit time of the source.
  epoch="$(bash tools/wheel-build/source_date_epoch.sh "${src}")"
  export "${epoch?}"
  PYTHON=python bash tools/wheel-build/tei_wheel.sh "${src}" "${src}/target/release/text-embeddings-router" \
    "${features}" aarch64-apple-darwin "${BACKEND}" "${PLATFORM}" out
  # Fails when any Mach-O needs a newer macOS than the tag (macosx_14_0) allows.
  delocate-wheel --require-archs arm64 --require-target-macos-version 14.0 -w out/repaired out/dist/*.whl
  python -m zipfile -l out/repaired/*.whl > wheel-listing.txt
  if grep -q '\.dylibs/' wheel-listing.txt; then echo "pack_tei: delocate grafted a library into the wheel" >&2; exit 1; fi
  ls -l out out/repaired
)

check_tag() {  # wheel
  local name info
  name="$(basename "$1")"
  case "${name}" in
    lilbee_engine_tei-"${WHEEL_VERSION}"-1."${BACKEND//-/_}"-py3-none-"${PLATFORM}".whl) ;;
    *)
      echo "pack_tei: wrong tag on ${name}: expected version ${WHEEL_VERSION}, backend ${BACKEND} and platform ${PLATFORM}" >&2
      return 1
      ;;
  esac
  info="$(dist_info "$1" WHEEL)" || return 1
  grep -qxF "Tag: py3-none-${PLATFORM}" <<< "${info}" \
    || { echo "pack_tei: the WHEEL file of ${name} lacks the tag py3-none-${PLATFORM}" >&2; return 1; }
}

# Install the wheel and start its router. A Linux router also starts on an emulated CPU: the CPU of a runner can have
# AVX-512, so a start on it cannot see a router built for it. Nehalem has SSE4.2 and no AVX, the same floor as the
# x86-64-v2 build; qemu stops the router with SIGILL on any newer instruction.
start_router() {  # wheel
  local root="${PWD}" router stubs="" cpu cpus=("")
  python -m venv check-venv
  # No pin: the wheel needs no other package, so pip installs this one local file.
  check-venv/bin/python -m pip install -q "$1"
  router="$(check-venv/bin/python -c "import lilbee_engine_tei as m, pathlib; print(pathlib.Path(m.__file__).parent / 'bin' / 'text-embeddings-router')")"
  echo "router: ${router}"
  if [ -d out/stubs ]; then stubs="${root}/out/stubs"; fi
  if [ "$(host_os)" = Linux ]; then cpus+=(Nehalem); fi
  for cpu in "${cpus[@]}"; do
    (cd / && EMULATE_CPU="${cpu}" STUB_DIR="${stubs}" bash "${root}/tools/wheel-build/smoke_tei.sh" "${router}" "${BACKEND}")
  done
}

main() {
  local wheel
  cd "$(dirname "${BASH_SOURCE[0]}")/../.."
  require_inputs
  use_build_python
  # A checkout that another umask made holds other modes.
  chmod -R u=rwX,go=rX "${WHEEL_TREE}"
  mkdir out
  if [ "$(host_os)" = Linux ]; then build_linux; else build_metal; fi
  collect_wheel out/repaired wheel
  cp out/license-expression.txt wheel/
  wheel="$(ls wheel/*.whl)"
  check_tag "${wheel}"
  python tools/wheel-build/check_tei_wheel.py "${wheel}" "${BACKEND}" wheel/license-expression.txt
  start_router "${wheel}"
  # Only a wheel that passed its checks reaches the output folder.
  mkdir -p "${OUT}"
  cp wheel/* "${OUT}/"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main
fi
