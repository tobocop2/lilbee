#!/usr/bin/env bash
# Build the llama.cpp engine pack of one cell. Run it from a fresh lilbee checkout:
#
#   OUT=/path/to/out BACKEND=cpu ... bash tools/wheel-build/pack_llama.sh
#
# The engine-packs repository sets the variables from its cells file (scripts/pack.sh):
#   UPSTREAM_REPO, UPSTREAM_TAG   the llama.cpp release; the pack workflow checks it against the lilbee pin
#   WHEEL_VERSION, BUILD_NUMBER   the version and the build number of the wheel
#   GO_LICENSES_VERSION           the go-licenses release
#   BACKEND, PLATFORM             the backend and the platform tag of the wheel
#   LICENSE_TERMS                 the licence terms of the cell, joined with " AND "
#   BUILD, MANYLINUX_IMAGE        BUILD=source compiles the server in the image; it is empty for the other cells
#   ASSET, EXTRA_ASSET            the archives that the other cells repack (EXTRA_ASSET may be empty)
#   MACOSX_DEPLOYMENT_TARGET      the macOS floor of the Go tools
#   OUT                           the folder that receives the wheel and its sha256 file
#
# engine-versions.env holds the pins of llama-swap, gguf-parser and Go. The script leaves license-expression.txt,
# cuda-libs.sha256 and vs-licence.docx in the checkout for the check of the wheel, and out/ beside them.
#
# The script needs Python 3.12, Go, git and a gh login. A source cell needs docker, another cell needs cmake. The build
# writes into the wheel project, so the script refuses a checkout that an earlier build used.

set -euo pipefail

# The mode of each file goes into the wheel.
umask 022

# shellcheck source=tools/wheel-build/pack_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/pack_common.sh"

WHEEL_TREE=packaging/engine-wheel
BIN="${WHEEL_TREE}/lilbee_engine_llama_cpp/bin"

require_inputs() {
  local os name names tools
  os="$(host_os)"
  names=(UPSTREAM_REPO UPSTREAM_TAG WHEEL_VERSION BUILD_NUMBER GO_LICENSES_VERSION BACKEND PLATFORM LICENSE_TERMS
    MACOSX_DEPLOYMENT_TARGET OUT)
  tools=(go git gh)
  if [ "${BUILD:-}" = source ]; then
    [ "${os}" = Linux ] || { echo "pack_llama: no source cell builds on ${os}" >&2; return 2; }
    names+=(MANYLINUX_IMAGE)
    tools+=(docker)
  else
    names+=(ASSET)
    tools+=(cmake)
  fi
  for name in "${names[@]}"; do
    [ -n "${!name:-}" ] || { echo "pack_llama: ${name} is not set" >&2; return 2; }
  done
  for name in "${tools[@]}"; do
    command -v "${name}" > /dev/null || { echo "pack_llama: ${name} is not installed" >&2; return 2; }
  done
  for name in upstream deps-build cuda-libs.sha256 license-expression.txt vs-licence.docx repaired out pack-venv \
    "${BIN}" "${WHEEL_TREE}/dist"; do
    [ ! -e "${name}" ] || { echo "pack_llama: an earlier build used this checkout (${name} exists). Use a fresh one." >&2; return 2; }
  done
}

# The compiler goes into the Go tools, so it is the pinned one. GOTOOLCHAIN=local stops a download of another one.
require_go() {
  local version
  export GOTOOLCHAIN=local
  version="$(go env GOVERSION | tr -d '\r')"
  [ "${version}" = "go${ENGINE_GO_VERSION}" ] \
    || { echo "pack_llama: Go ${ENGINE_GO_VERSION} is required, this computer has ${version}" >&2; return 2; }
}

fetch_archives() {
  local name
  for name in "${ASSET}" ${EXTRA_ASSET:+"${EXTRA_ASSET}"}; do
    fetch_release_asset "${UPSTREAM_REPO}" "${UPSTREAM_TAG}" "${name}" upstream
  done
}

# Build the server with CUDA. The licence step reads the source trees that the configure step of this build fetched.
build_cuda() {
  docker run --rm -v "${PWD}:/work" -w /work \
    -e UPSTREAM_REPO -e UPSTREAM_TAG \
    -e "BIN_DIR=/work/${BIN}" \
    -e SHA_FILE=/work/cuda-libs.sha256 -e DEPS_DIR=/work/deps-build/_deps -e "OWNER=$(id -u):$(id -g)" \
    "${MANYLINUX_IMAGE}" \
    bash /work/tools/wheel-build/llama_cuda_manylinux.sh
  ls -l "${BIN}" deps-build/_deps
  cat cuda-libs.sha256
}

# Build the server for CPU, Vulkan or ROCm. A ROCm build also writes LICENSE-ROCm-*.
build_gpu() {
  docker run --rm -v "${PWD}:/work" -w /work \
    -e BACKEND -e UPSTREAM_REPO -e UPSTREAM_TAG \
    -e "BIN_DIR=/work/${BIN}" \
    -e "LICENSE_DIR=/work/${WHEEL_TREE}" \
    -e "OWNER=$(id -u):$(id -g)" \
    "${MANYLINUX_IMAGE}" \
    bash /work/tools/wheel-build/llama_gpu_manylinux.sh
  ls -l "${BIN}"
  df -h / | tail -1
}

# Build llama-swap and gguf-parser, and copy their licence files.
build_go_tools() {
  local bin="${PWD}/${BIN}" suffix=""
  if [ "$(host_os)" = Windows ]; then suffix=.exe; fi
  git clone -q --depth 1 --branch "${ENGINE_LLAMA_SWAP_VERSION}" https://github.com/mostlygeek/llama-swap.git "${WORK}/llama-swap"
  (cd "${WORK}/llama-swap" && go build -trimpath -o "${bin}/llama-swap${suffix}" .)
  cp "${WORK}/llama-swap/LICENSE.md" "${WHEEL_TREE}/LICENSE-llama-swap"
  git clone -q --depth 1 --branch "${ENGINE_GGUF_PARSER_REF}" "${ENGINE_GGUF_PARSER_REPO}" "${WORK}/gguf-parser-go"
  # The parser reports the version in main.Version. Without the stamp it reports v0.0.0.
  (cd "${WORK}/gguf-parser-go/cmd/gguf-parser" && go build -trimpath -ldflags "-X main.Version=${ENGINE_GGUF_PARSER_REF}" -o "${bin}/gguf-parser${suffix}" .)
  cp "${WORK}/gguf-parser-go/LICENSE" "${WHEEL_TREE}/LICENSE-gguf-parser"
}

golicenses() {
  go run "github.com/google/go-licenses/v2@${GO_LICENSES_VERSION}" "$@"
}

# Fail on a copyleft or unknown licence, save the texts of the modules that the binary links, and add their SPDX names
# to names.txt.
collect_modules() {  # name folder [extra ignore arguments]
  local name="$1" dir="$2" save="${PWD}/${WHEEL_TREE}/THIRD-PARTY/$1"
  shift 2
  (
    cd "${dir}"
    golicenses check . "$@" --disallowed_types=forbidden,restricted,reciprocal,unknown
    golicenses save . "$@" --save_path "${save}"
    golicenses report . "$@" --template "${WORK}/names.tmpl" >> "${WORK}/names.txt"
  )
}

collect_go_licences() {
  local std pkg goroot ignore=()
  # go-licenses cannot read the standard library, so every standard package is ignored. The licence of the library is
  # copied from GOROOT.
  std="$(go list std | tr -d '\r')"
  while IFS= read -r pkg; do ignore+=(--ignore "${pkg}"); done <<< "${std}"
  printf '{{ range . }}{{ .LicenseName }}\n{{ end }}' > "${WORK}/names.tmpl"
  collect_modules llama-swap "${WORK}/llama-swap" "${ignore[@]}"
  # The binary lives in a nested module that has no licence file of its own; the licence file at the repository root covers it.
  collect_modules gguf-parser "${WORK}/gguf-parser-go/cmd/gguf-parser" "${ignore[@]}" --ignore github.com/gpustack/gguf-parser-go/cmd/gguf-parser
  goroot="$(go env GOROOT | tr -d '\r')"
  cp "${goroot//\\//}/LICENSE" "${WHEEL_TREE}/LICENSE-Go"
}

# Collect the licences of llama.cpp and the code it carries.
collect_llama_licences() {
  local args=(--skip tools/ui) deps="" source repo tag
  git clone -q --depth 1 --branch "${UPSTREAM_TAG}" "https://github.com/${UPSTREAM_REPO}.git" "${WORK}/llama.cpp"
  # A source cell sets LLAMA_OPENSSL=OFF and LLAMA_BUILD_UI=OFF, so it links no fetched TLS code and no web UI.
  if [ "${BUILD:-}" != source ]; then
    # Configure the tag with the BoringSSL flag that the four release jobs share, and compile nothing. The collector
    # reads the licence files of what that fetched.
    cmake -S "${WORK}/llama.cpp" -B "${WORK}/deps-build" -DLLAMA_BUILD_BORINGSSL=ON \
      -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_NATIVE=OFF
    args=()
    deps="${WORK}/deps-build/_deps"
    if [ "${BACKEND}" = cuda ]; then
      # The CUDA asset compiles in CCCL; a configure with CUDA needs the toolkit, which the build host lacks.
      install_build_tools python tools/wheel-build/llama-tools-windows.constraints
      source="$(python tools/wheel-build/cccl_source.py "${WORK}/llama.cpp" "${ASSET}" "${UPSTREAM_TAG}" | tr -d '\r')"
      read -r repo tag <<< "${source}"
      git -c core.longpaths=true clone -q --depth 1 --branch "${tag}" "${repo}" "${deps}/cccl-src"
      echo "CCCL ${tag} from ${repo}"
    fi
  elif [ "${BACKEND}" = cuda ]; then
    # build_cuda copied the trees that its configure fetched (CCCL) here.
    deps="${PWD}/deps-build/_deps"
  fi
  bash tools/wheel-build/collect_vendored_licences.sh ${args[@]+"${args[@]}"} \
    "${WORK}/llama.cpp" "${PWD}/${WHEEL_TREE}/THIRD-PARTY/llama.cpp" "${WORK}/vendored-terms.txt" "${deps}"
}

# Set LICENSE_EXPRESSION: the terms of the cell, then the Go module licences, the Go library licence and the vendored
# llama.cpp licences.
name_licences() {
  local file newline=$'\n'
  LICENSE_EXPRESSION="$({
    printf '%s\n' "${LICENSE_TERMS// AND /${newline}}"
    { cat "${WORK}/names.txt" "${WORK}/vendored-terms.txt"; echo BSD-3-Clause; } | tr -d '\r' | sort -u
  } | merge_terms)"
  echo "License-Expression: ${LICENSE_EXPRESSION}"
  printf '%s\n' "${LICENSE_EXPRESSION}" > license-expression.txt
  for file in "${WHEEL_TREE}"/LICENSE-*; do
    [ -s "${file}" ] || { echo "pack_llama: empty licence file ${file}" >&2; return 1; }
  done
  find "${WHEEL_TREE}/THIRD-PARTY" -type f | wc -l
}

# Build the Go tools, collect every licence that they and llama.cpp carry, and set LICENSE_EXPRESSION.
build_tools_and_licences() {
  # The locale sets the order of the list of notices in the wheel. The macOS packs list them in the English order of
  # the macOS runner, the other packs in the order of the bytes.
  local -x LC_ALL=C
  if [ "$(host_os)" = macOS ]; then LC_ALL=en_US.UTF-8; fi
  gh api -H "Accept: application/vnd.github.raw" "repos/${UPSTREAM_REPO}/contents/LICENSE?ref=${UPSTREAM_TAG}" > "${WHEEL_TREE}/LICENSE-llama.cpp"
  build_go_tools
  collect_go_licences
  collect_llama_licences
  name_licences
}

# Put the Visual C++ runtime beside the exe.
add_windows_runtime() {
  bash tools/wheel-build/bundle_windows_runtime.sh "${BIN}" "${WHEEL_TREE}/NOTICE-MICROSOFT-VC-RUNTIME" vs-licence.docx
  # This build uses libomp.dll, so nothing needs vcomp140.
  if (cd "${BIN}" && grep -rilaF vcomp140 --include='*.dll' --include='*.exe' . | grep -vi '^./vcomp140.dll$'); then
    echo "pack_llama: a binary imports vcomp140.dll" >&2
    return 1
  fi
  rm -f packaging/engine-wheel/lilbee_engine_llama_cpp/bin/vcomp140.dll
}

# Set the version, the licence expression and the licence files in the wheel project, and write the pin of the engine.
stamp_project() {
  (cd "${WHEEL_TREE}" && python - "${WHEEL_VERSION}" "${LICENSE_EXPRESSION}" \
    "llama-cpp-${UPSTREAM_TAG}+swap-${ENGINE_LLAMA_SWAP_VERSION}+gguf-${ENGINE_GGUF_PARSER_REF}" <<'PY'
import pathlib, re, sys
version, expression, pin = sys.argv[1:]
p = pathlib.Path("pyproject.toml")
text = re.sub(r'(?m)^version = ".*"$', f'version = "{version}"', p.read_text(), count=1)
# License-Expression in METADATA needs setuptools 77 or later; lilbee refuses a wheel without it.
text = text.replace('requires = ["setuptools>=68"]', 'requires = ["setuptools>=77"]', 1)
files = '["LICENSE-*", "NOTICE-*", "THIRD-PARTY/**"]'
text = text.replace('requires-python', 'license = "%s"\nlicense-files = %s\nrequires-python' % (expression, files), 1)
p.write_text(text)
# The text of this file is in the wheel, so it keeps the words that the published packs have.
pathlib.Path("lilbee_engine_llama_cpp/pins.py").write_text(
    '"""Engine build identity; written by the build-packs workflow."""\n\nENGINE_PIN = "%s"\n' % pin)
PY
  )
  grep -E '^(license|license-files|requires)' "${WHEEL_TREE}/pyproject.toml"
  cat "${WHEEL_TREE}/lilbee_engine_llama_cpp/pins.py"
}

# Build the wheel into the dist folder, and on macOS its repaired copy into repaired. The subshell keeps the build
# variables here.
build_wheel() (
  # The wheel build, the retag and delocate stamp every entry with the commit time of the llama.cpp tag.
  epoch="$(bash tools/wheel-build/source_date_epoch.sh "${WORK}/llama.cpp")"
  export "${epoch?}"
  # A relative path: Windows Python does not read a Git Bash path.
  files=(tools/wheel-build/build-tools.constraints)
  if [ "$(host_os)" = macOS ]; then files+=(tools/wheel-build/llama-tools-macos.constraints); fi
  install_build_tools python "${files[@]}"
  (
    cd "${WHEEL_TREE}"
    build_wheel_with_tools python -C--build-option="--build-number=${BUILD_NUMBER}.${BACKEND}"
    python -m wheel tags --remove --platform-tag "${PLATFORM}" dist/*.whl
    ls dist
  )
  if [ "$(host_os)" = macOS ]; then repair_macos; fi
)

repair_macos() {
  local wheel
  # Fails when any Mach-O needs a newer macOS than the tag (macosx_14_0) allows.
  delocate-wheel --require-archs arm64 --require-target-macos-version 14.0 -w repaired "${WHEEL_TREE}"/dist/*.whl
  ls repaired
  for wheel in repaired/*.whl; do
    case "${wheel}" in *-macosx_14_0_arm64.whl) ;; *) echo "pack_llama: wrong tag on ${wheel}" >&2; return 1 ;; esac
    python -m zipfile -l "${wheel}" > wheel-listing.txt
    if grep -q '\.dylibs/' wheel-listing.txt; then echo "pack_llama: delocate grafted a library into ${wheel}" >&2; return 1; fi
  done
}

# Write the list of gfx targets that the rocBLAS kernels in the pack cover.
list_gfx_targets() {  # file
  local kernels name
  for kernels in "${BIN}"/rocblas/library/TensileLibrary_lazy_gfx*.dat; do
    [ -e "${kernels}" ] || continue
    name="${kernels##*/TensileLibrary_lazy_}"
    echo "${name%.dat}"
  done | sort -u > "$1"
  [ -s "$1" ] || { echo "pack_llama: the pack carries no rocBLAS kernels" >&2; return 1; }
  cat "$1"
}

main() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.."
  require_inputs
  # shellcheck source=/dev/null
  . ./engine-versions.env
  require_go
  use_build_python
  # A checkout that another umask made holds other modes.
  if [ "$(host_os)" != Windows ]; then chmod -R u=rwX,go=rX "${WHEEL_TREE}"; fi
  WORK="$(mktemp -d)"
  mkdir -p "${BIN}"
  if [ "${BUILD:-}" != source ]; then fetch_archives; elif [ "${BACKEND}" = cuda ]; then build_cuda; fi
  build_tools_and_licences
  if [ "${BACKEND}" = cuda ]; then
    python tools/wheel-build/llama_cuda_notice.py "${WHEEL_TREE}/NOTICE-NVIDIA-CUDA"
    wc -c "${WHEEL_TREE}/NOTICE-NVIDIA-CUDA"
  fi
  if [ "${BUILD:-}" != source ]; then
    python tools/wheel-build/repack_llama_archives.py "${BIN}" "upstream/${ASSET}" ${EXTRA_ASSET:+"upstream/${EXTRA_ASSET}"}
    ls -l "${BIN}"
  elif [ "${BACKEND}" != cuda ]; then
    build_gpu
  fi
  if [ "$(host_os)" = Windows ]; then add_windows_runtime; fi
  stamp_project
  build_wheel
  if [ "$(host_os)" = macOS ]; then collect_wheel repaired out; else collect_wheel "${WHEEL_TREE}/dist" out; fi
  if [ "${BACKEND}" = rocm ]; then list_gfx_targets out/lilbee-linux-x86_64-rocm.gfx.txt; fi
  mkdir -p "${OUT}"
  cp out/* "${OUT}/"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main
fi
