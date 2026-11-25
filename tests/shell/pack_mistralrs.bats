#!/usr/bin/env bats
# tools/wheel-build/pack_mistralrs.sh in a copy of the tree, with stubs for gh, git, cargo-about and the
# Python build tools. The licence, notice and repack scripts are the real ones. The wheel check is a stub
# that records its arguments: tests/test_mistralrs_pack_tools.py tests the check.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  ROOT="${BATS_TEST_TMPDIR}/lilbee"
  BIN="${BATS_TEST_TMPDIR}/bin"
  mkdir -p "${ROOT}/packaging" "${ROOT}/tools" "${BIN}" "${BATS_TEST_TMPDIR}/assets"
  cp -R "${REPO_ROOT}/tools/wheel-build" "${ROOT}/tools/"
  cp -R "${REPO_ROOT}/packaging/engine-wheel-mistral-rs" "${ROOT}/packaging/"
  SCRIPT="${ROOT}/tools/wheel-build/pack_mistralrs.sh"
  TREE="${ROOT}/packaging/engine-wheel-mistral-rs"
  CHECK_LOG="${BATS_TEST_TMPDIR}/check.log"
  printf '%s\n' "printf '%s\n' \"\$@\" > '${CHECK_LOG}'" '[ -z "${CHECK_FAIL-}" ]' > "${ROOT}/tools/wheel-build/check_mistralrs_wheel.sh"

  REAL_PYTHON="$(command -v python3)"
  cp "${BATS_TEST_DIRNAME}/stubs/gh-pack" "${BIN}/gh"
  cp "${BATS_TEST_DIRNAME}/stubs/git-pack" "${BIN}/git"
  cp "${BATS_TEST_DIRNAME}/stubs/python-pack" "${BIN}/python"
  chmod +x "${BIN}"/*
  PATH="${BIN}:${PATH}"

  export REAL_PYTHON GH_ASSETS="${BATS_TEST_TMPDIR}/assets" GH_LOG="${BATS_TEST_TMPDIR}/gh.log"
  export GIT_LOG="${BATS_TEST_TMPDIR}/git.log" ABOUT_LOG="${BATS_TEST_TMPDIR}/about.log"
  export PYTHON_LOG="${BATS_TEST_TMPDIR}/python.log"
  export CARGO_HOME="${BATS_TEST_TMPDIR}/cargo"
  export MISTRALRS_REPO=owner/mistral.rs MISTRALRS_TAG=v0.9.5 ASSET=m.tar.gz BACKEND=cpu
  export PLATFORM=manylinux_2_35_x86_64 RUST_TARGET=x86_64-unknown-linux-gnu RUST_FEATURES=
  export CARGO_ABOUT_VERSION=0.9.3 ABOUT_TARGET=x86_64-unknown-linux-musl MISTRALRS_CHAT_MODEL=m
  export RUNNER_OS=Linux OUT="${BATS_TEST_TMPDIR}/packs-out"
  unset SOURCE_DATE_EPOCH

  mkdir -p "${BATS_TEST_TMPDIR}/make/cargo-about-0.9.3"
  echo tool > "${BATS_TEST_TMPDIR}/make/mistralrs"
  touch -t 202311142213.20 "${BATS_TEST_TMPDIR}/make/mistralrs"
  TZ=UTC tar -czf "${GH_ASSETS}/m.tar.gz" -C "${BATS_TEST_TMPDIR}/make" mistralrs
  EPOCH="$("${REAL_PYTHON}" -c 'import os, sys; print(int(os.stat(sys.argv[1]).st_mtime))' "${BATS_TEST_TMPDIR}/make/mistralrs")"
  cp "${BATS_TEST_DIRNAME}/stubs/cargo-about-pack" "${BATS_TEST_TMPDIR}/make/cargo-about-0.9.3/cargo-about"
  chmod +x "${BATS_TEST_TMPDIR}/make/cargo-about-0.9.3/cargo-about"
  tar -czf "${GH_ASSETS}/cargo-about-0.9.3-x86_64-unknown-linux-musl.tar.gz" -C "${BATS_TEST_TMPDIR}/make" cargo-about-0.9.3
  WHEEL=lilbee_engine_mistral_rs-0.9.5-1.cpu-py3-none-manylinux_2_35_x86_64.whl
  cd "${BATS_TEST_TMPDIR}"
}

# Source the script in the tree and run one of its functions, as main does.
stage() {
  (cd "${ROOT}" && bash -c 'set -euo pipefail; . tools/wheel-build/pack_mistralrs.sh; "$@"' stage "$@")
}

# Fail unless pip, the build and the retag ran, each with the interpreter in that folder of the venv.
built_in_the_venv() {  # folder
  local calls outside
  calls="$(grep -E '^-m (pip|build|wheel) ' "${PYTHON_LOG}")"
  [ "$(wc -l <<< "${calls}")" -eq 3 ]
  outside="$(grep -vF " exe=${ROOT}/pack-venv/$1/python " <<< "${calls}" || true)"
  [ -z "${outside}" ] || { echo "ran outside the venv: ${outside}"; return 1; }
}

@test "the wheel and its sha256 file reach the output folder" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(ls "${OUT}")" = "${WHEEL}
${WHEEL}.sha256" ]
  (cd "${OUT}" && sha256sum -c "${WHEEL}.sha256")
}

@test "every variable of the contract is required" {
  local name
  for name in MISTRALRS_REPO MISTRALRS_TAG ASSET BACKEND PLATFORM RUST_TARGET CARGO_ABOUT_VERSION ABOUT_TARGET \
    MISTRALRS_CHAT_MODEL OUT; do
    run env "${name}=" bash "${SCRIPT}"
    [ "$status" -eq 2 ]
    [[ "$output" == *"${name} is not set"* ]]
  done
  [ ! -e "${GH_LOG}" ]
}

@test "a checkout that an earlier build used is refused" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  run bash "${SCRIPT}"
  [ "$status" -eq 2 ]
  [[ "$output" == *"an earlier build used this checkout"* ]]
}

@test "a Python other than 3.12 is refused before a download" {
  local name
  for name in python3.12 python3; do cp "${BIN}/python" "${BIN}/${name}"; done
  REAL_PYTHON="${REAL_PYTHON}" PYTHON_VERSION=3.13 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"Python 3.12 is not on PATH"* ]]
  [ ! -e "${GH_LOG}" ]
}

@test "the upstream archive and cargo-about are fetched from the pinned releases" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  grep -qx 'release download v0.9.5 -R owner/mistral.rs --pattern m.tar.gz --dir upstream' "${GH_LOG}"
  grep -qx 'release download 0.9.3 -R EmbarkStudios/cargo-about --pattern cargo-about-0.9.3-x86_64-unknown-linux-musl.tar.gz --dir cargo-about' "${GH_LOG}"
  grep -qx 'clone -q --depth 1 --branch v0.9.5 https://github.com/owner/mistral.rs.git mistral.rs' "${GIT_LOG}"
}

@test "a digest that differs stops the build" {
  GH_DIGEST=sha256:0 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"digest mismatch m.tar.gz"* ]]
  [ ! -e "${OUT}" ]
}

@test "a cargo-about digest that differs stops the build before cargo-about runs" {
  local name=cargo-about-0.9.3-x86_64-unknown-linux-musl.tar.gz
  GH_DIGEST=sha256:0 GH_DIGEST_OF="${name}" run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"digest mismatch ${name}"* ]]
  [ ! -e "${ABOUT_LOG}" ]
  [ ! -e "${OUT}" ]
}

@test "cargo-about reads the locked graph of the target and of the features" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(cat "${ABOUT_LOG}")" = "generate --format json --locked -c ../tools/wheel-build/mistralrs-about.toml -m mistralrs-cli/Cargo.toml --target x86_64-unknown-linux-gnu -o ../about.json" ]
}

@test "the features of the cell go to cargo-about" {
  RUST_FEATURES=metal run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  grep -q -- '--target x86_64-unknown-linux-gnu --features metal -o' "${ABOUT_LOG}"
}

@test "a cargo-about failure stops the build" {
  ABOUT_FAIL=1 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [ ! -e "${OUT}" ]
}

@test "the licences and the notices are in the wheel project before the build starts" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  local file
  for file in LICENSE-mistral.rs NOTICE-OpenSSL SOURCE-MPL-2.0 THIRD-PARTY/mistral.rs/MIT-001.txt \
    THIRD-PARTY/vendored/mistral.rs__sub__NOTICE lilbee_engine_mistral_rs/bin/mistralrs; do
    grep -qx "./${file}" "${PYTHON_LOG}.tree"
  done
}

@test "the build and the retag receive the time of the tool in the archive" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  grep -q "^-m build --wheel .*| epoch=${EPOCH} " "${PYTHON_LOG}"
  grep -q "^-m wheel tags .*| epoch=${EPOCH} " "${PYTHON_LOG}"
}

@test "a time that the shell already holds is replaced" {
  SOURCE_DATE_EPOCH=1 run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  grep -q "^-m build --wheel .*| epoch=${EPOCH} " "${PYTHON_LOG}"
}

@test "an archive without the tool stops the build" {
  tar -czf "${GH_ASSETS}/m.tar.gz" -C "${BATS_TEST_TMPDIR}/make" cargo-about-0.9.3
  run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [ ! -e "${OUT}" ]
}

@test "pip installs the pinned build tools by their digests, and the build installs nothing" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  grep -q '^-m pip install --require-hashes --force-reinstall -r ../../tools/wheel-build/build-tools.constraints -r ../../tools/wheel-build/windows-bundle-tools.constraints | .* cwd=engine-wheel-mistral-rs$' "${PYTHON_LOG}"
  grep -q '^-m build --wheel --no-isolation .* cwd=engine-wheel-mistral-rs$' "${PYTHON_LOG}"
  [ "$(grep -c '^-m pip ' "${PYTHON_LOG}")" -eq 1 ]
}

@test "the stand-in for pip refuses what the pip of Python 3.12 refuses" {
  local file="${ROOT}/tools/wheel-build/build-tools.constraints"
  run "${BIN}/python" -m pip install --require-hashes -r "${file}"
  [ "$status" -eq 0 ]
  PIP_CONSTRAINT="${file}" run "${BIN}/python" -m pip install build wheel
  [ "$status" -eq 1 ]
  run "${BIN}/python" -m pip install -r "${file}"
  [ "$status" -eq 1 ]
  run "${BIN}/python" -m pip install --require-hashes -r "${ROOT}/no-such-file"
  [ "$status" -eq 1 ]
  cd "${TREE}"
  run "${BIN}/python" -m build --wheel
  [ "$status" -eq 1 ]
  [ ! -e dist ]
}

@test "the constraints pin every build tool to one version and one digest" {
  local file="${REPO_ROOT}/tools/wheel-build/build-tools.constraints" name
  # build runs `python -m build`. It needs packaging and pyproject_hooks, and colorama on Windows.
  # setuptools is the build backend of the wheel project. wheel runs `python -m wheel tags`, and it needs packaging.
  for name in build colorama packaging pyproject_hooks setuptools wheel; do
    grep -qE "^${name}==[0-9.]+ \\\\$" "${file}" || { echo "no pinned ${name} in ${file}"; return 1; }
  done
  [ "$(grep -cE '^[a-z_]+==' "${file}")" -eq 6 ]
  [ "$(grep -cE '^    --hash=sha256:[0-9a-f]{64}$' "${file}")" -eq 6 ]
}

@test "pefile for the Windows packs is pinned in its own file, apart from the build tools" {
  local check="${REPO_ROOT}/tools/wheel-build/windows-bundle-tools.constraints"
  grep -qE '^pefile==[0-9.]+ \\$' "${check}"
  [ "$(grep -cE '^[a-z_]+==' "${check}")" -eq 1 ]
  [ "$(grep -cE '^    --hash=sha256:[0-9a-f]{64}$' "${check}")" -eq 1 ]
  ! grep -q '^pefile' "${REPO_ROOT}/tools/wheel-build/build-tools.constraints"
}

@test "the wheel project needs only the setuptools of the constraints" {
  local file="${REPO_ROOT}/tools/wheel-build/build-tools.constraints" project="${REPO_ROOT}/packaging/engine-wheel-mistral-rs/pyproject.toml"
  # The build runs without isolation, so the constraints must also pin a new entry of this list.
  grep -qxF 'requires = ["setuptools>=77"]' "${project}"
  grep -qxF 'build-backend = "setuptools.build_meta"' "${project}"
  # The WHEEL files of the baseline packs name this version. Another version changes the bytes of every pack.
  grep -qxF 'setuptools==84.2.1 \' "${file}"
}

@test "a venv that the checkout already holds stops the build before pip runs" {
  mkdir "${ROOT}/pack-venv"
  run bash "${SCRIPT}"
  [ "$status" -eq 2 ]
  [[ "$output" == *"an earlier build used this checkout (pack-venv exists)"* ]]
  [ ! -e "${PYTHON_LOG}" ]
}

@test "the build number holds the backend and the retag the platform" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  grep -q '^-m build --wheel --no-isolation -C--build-option=--build-number=1.cpu ' "${PYTHON_LOG}"
  grep -q '^-m wheel tags --remove --platform-tag manylinux_2_35_x86_64 dist/' "${PYTHON_LOG}"
}

for cc in 80 86 89 90 100 120; do
  @test "the build number of the cuda-sm${cc} cell is 1.cuda_sm${cc}, with no hyphen" {
    BACKEND="cuda-sm${cc}" run stage build_wheel
    [ "$status" -eq 0 ]
    grep -q -- "--build-number=1.cuda_sm${cc} " "${PYTHON_LOG}"
  }
done

@test "the build number of the Metal cell stays 1.metal" {
  BACKEND=metal run stage build_wheel
  [ "$status" -eq 0 ]
  grep -q -- '--build-number=1.metal ' "${PYTHON_LOG}"
}

@test "the build runs with umask 022 whatever the shell has" {
  umask 0077
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  grep -q '^-m build --wheel .* umask=0022 ' "${PYTHON_LOG}"
  [ "$(find "${TREE}" -name 'NOTICE-OpenSSL' -perm 644 | wc -l)" -eq 1 ]
  [ "$(find "${TREE}/lilbee_engine_mistral_rs/bin/mistralrs" -perm 755 | wc -l)" -eq 1 ]
}

@test "pip, the build and the retag run in the venv of the checkout" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(grep -c '^-m venv pack-venv ' "${PYTHON_LOG}")" -eq 1 ]
  built_in_the_venv bin
}

@test "the project file holds the version, the expression and the licence files" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  grep -qx 'version = "0.9.5"' "${TREE}/pyproject.toml"
  grep -qx 'license = "MIT AND Apache-2.0 AND MPL-2.0 AND OpenSSL AND BSD-3-Clause"' "${TREE}/pyproject.toml"
  grep -qxF 'license-files = ["LICENSE-*", "NOTICE-*", "SOURCE-*", "THIRD-PARTY/**"]' "${TREE}/pyproject.toml"
}

@test "the check receives the wheel, the crate licences and the expression" {
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(cat "${CHECK_LOG}")" = "out/${WHEEL}
crate-licences.txt
MIT AND Apache-2.0 AND MPL-2.0 AND OpenSSL AND BSD-3-Clause" ]
}

@test "a wheel that fails its check does not reach the output folder" {
  CHECK_FAIL=1 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [ ! -e "${OUT}" ]
}

@test "two wheels stop the build" {
  BUILD_WHEELS=2 run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"expected one wheel"* ]]
  [ ! -e "${OUT}" ]
}

@test "Windows adds the runtime and its licence term; the other systems do not" {
  local marker="${BATS_TEST_TMPDIR}/runtime"
  printf '%s\n' "echo \"\$@\" > '${marker}'" 'touch "$1/vcomp140.dll"; echo notice > "$2"' > "${ROOT}/tools/wheel-build/bundle_windows_runtime.sh"
  mkdir -p "${TREE}/lilbee_engine_mistral_rs/bin"
  echo exe > "${TREE}/lilbee_engine_mistral_rs/bin/mistralrs.exe"
  run stage add_windows_runtime
  [ "$status" -eq 0 ]
  [ ! -e "${marker}" ]
  echo MIT > "${ROOT}/crate-licences.txt"
  run stage name_licences
  [ "$output" = "License-Expression: MIT AND Apache-2.0 AND BSD-3-Clause" ]

  export RUNNER_OS=Windows
  run stage add_windows_runtime
  [ "$status" -eq 0 ]
  [ "$(cat "${marker}")" = "packaging/engine-wheel-mistral-rs/lilbee_engine_mistral_rs/bin packaging/engine-wheel-mistral-rs/NOTICE-MICROSOFT-VC-RUNTIME" ]
  run stage name_licences
  [ "$output" = "License-Expression: MIT AND Apache-2.0 AND BSD-3-Clause AND LicenseRef-Microsoft-VC-Runtime" ]
}

# A Windows cell: the bundler stand-in adds two runtime DLLs, and the prune stand-in deletes vcomp140 and records its arguments.
windows_cell() {
  export RUNNER_OS=Windows PLATFORM=win_amd64 ASSET=m.zip ABOUT_TARGET=x86_64-pc-windows-msvc
  local about=cargo-about-0.9.3-x86_64-pc-windows-msvc
  mkdir -p "${BATS_TEST_TMPDIR}/make/${about}" "${ROOT}/tools/qa"
  cp "${BATS_TEST_DIRNAME}/stubs/cargo-about-pack" "${BATS_TEST_TMPDIR}/make/${about}/cargo-about.exe"
  chmod +x "${BATS_TEST_TMPDIR}/make/${about}/cargo-about.exe"
  tar -czf "${GH_ASSETS}/${about}.tar.gz" -C "${BATS_TEST_TMPDIR}/make" "${about}"
  echo exe > "${BATS_TEST_TMPDIR}/make/mistralrs.exe"
  (cd "${BATS_TEST_TMPDIR}/make" && "${REAL_PYTHON}" -m zipfile -c "${GH_ASSETS}/m.zip" mistralrs.exe)
  printf '%s\n' 'set -e' '[ -f "$1/mistralrs.exe" ]' 'echo dll > "$1/vcruntime140.dll"' 'echo dll > "$1/vcomp140.dll"' 'echo notice > "$2"' > "${ROOT}/tools/wheel-build/bundle_windows_runtime.sh"
  printf '%s\n' 'import os, sys' \
    "open('${BATS_TEST_TMPDIR}/prune.log', 'a').write(' '.join(sys.argv[1:]) + '\n')" \
    'os.remove(os.path.join(sys.argv[2], "vcomp140.dll"))' > "${ROOT}/tools/qa/assert_engine_bundle.py"
}

@test "the Windows cell builds with the runtime and its notice in the wheel project" {
  local wheel=lilbee_engine_mistral_rs-0.9.5-1.cpu-py3-none-win_amd64.whl
  windows_cell
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  grep -qx ./NOTICE-MICROSOFT-VC-RUNTIME "${PYTHON_LOG}.tree"
  grep -qx ./lilbee_engine_mistral_rs/bin/vcruntime140.dll "${PYTHON_LOG}.tree"
  grep -qx ./lilbee_engine_mistral_rs/bin/mistralrs.exe "${PYTHON_LOG}.tree"
  grep -qx 'license = "MIT AND Apache-2.0 AND MPL-2.0 AND OpenSSL AND BSD-3-Clause AND LicenseRef-Microsoft-VC-Runtime"' "${TREE}/pyproject.toml"
  built_in_the_venv Scripts
  [ "$(ls "${OUT}")" = "${wheel}
${wheel}.sha256" ]
}

@test "a runtime that cannot be bundled stops the whole Windows build" {
  export RUNNER_OS=Windows
  printf 'exit 1\n' > "${ROOT}/tools/wheel-build/bundle_windows_runtime.sh"
  run bash "${SCRIPT}"
  [ "$status" -eq 1 ]
  [ ! -e "${OUT}" ]
  [ ! -e "${PYTHON_LOG}.tree" ]
}

@test "the Windows build prunes the runtime DLLs that nothing imports before it builds the wheel" {
  windows_cell
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ "$(cat "${BATS_TEST_TMPDIR}/prune.log")" = "--prune-runtime lilbee_engine_mistral_rs/bin" ]
  grep -qx ./lilbee_engine_mistral_rs/bin/vcruntime140.dll "${PYTHON_LOG}.tree"
  ! grep -q vcomp140 "${PYTHON_LOG}.tree"
}

@test "a build that is not for Windows prunes nothing" {
  mkdir -p "${ROOT}/tools/qa"
  printf '%s\n' "open('${BATS_TEST_TMPDIR}/prune.log', 'w')" > "${ROOT}/tools/qa/assert_engine_bundle.py"
  run bash "${SCRIPT}"
  [ "$status" -eq 0 ]
  [ ! -e "${BATS_TEST_TMPDIR}/prune.log" ]
}

@test "a prune that fails stops the Windows build before the wheel is built" {
  windows_cell
  printf '%s\n' 'raise SystemExit(3)' > "${ROOT}/tools/qa/assert_engine_bundle.py"
  run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [ ! -e "${OUT}" ]
  [ ! -e "${PYTHON_LOG}.tree" ]
}

@test "a runtime that cannot be bundled stops the build" {
  printf 'exit 1\n' > "${ROOT}/tools/wheel-build/bundle_windows_runtime.sh"
  RUNNER_OS=Windows run stage add_windows_runtime
  [ "$status" -eq 1 ]
}

@test "an empty licence file stops the build" {
  local name
  echo MIT > "${ROOT}/crate-licences.txt"
  for name in LICENSE-mistral.rs NOTICE-OpenSSL SOURCE-MPL-2.0; do echo text > "${TREE}/${name}"; done
  run stage name_licences
  [ "$status" -eq 0 ]
  for name in LICENSE-mistral.rs NOTICE-OpenSSL SOURCE-MPL-2.0; do
    : > "${TREE}/${name}"
    run stage name_licences
    [ "$status" -eq 1 ]
    [[ "$output" == *"empty licence file packaging/engine-wheel-mistral-rs/${name}"* ]]
    echo text > "${TREE}/${name}"
  done
}

@test "a repack that prints no time stops the build" {
  printf 'print("nothing")\n' > "${ROOT}/tools/wheel-build/repack_mistralrs.py"
  run stage repack
  [ "$status" -eq 1 ]
  [[ "$output" == *"no time of the tool"* ]]
}
