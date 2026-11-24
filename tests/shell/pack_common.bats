#!/usr/bin/env bats
# tools/wheel-build/pack_common.sh: the functions that the pack scripts share.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  BIN="${BATS_TEST_TMPDIR}/bin"
  mkdir -p "${BIN}" "${BATS_TEST_TMPDIR}/assets"
  cp "${BATS_TEST_DIRNAME}/stubs/gh-pack" "${BIN}/gh"
  chmod +x "${BIN}/gh"
  PATH="${BIN}:${PATH}"
  export GH_ASSETS="${BATS_TEST_TMPDIR}/assets" GH_LOG="${BATS_TEST_TMPDIR}/gh.log"
  echo archive > "${GH_ASSETS}/a.tar.gz"
  echo other > "${GH_ASSETS}/b.tar.gz"
  # The scripts that source the file set these options.
  # shellcheck source=tools/wheel-build/pack_common.sh
  . "${REPO_ROOT}/tools/wheel-build/pack_common.sh"
  cd "${BATS_TEST_TMPDIR}"
}

# Run a function in a shell with the options that the pack scripts set.
strict() {
  bash -c 'set -euo pipefail; . "$0"; "$@"' "${REPO_ROOT}/tools/wheel-build/pack_common.sh" "$@"
}

# A PATH that holds the tools the functions need and no sha256sum.
without_sha256sum() {
  local tool
  mkdir -p "${BATS_TEST_TMPDIR}/bare"
  for tool in bash dirname basename shasum cat cp ls mkdir tr grep awk; do
    ln -s "$(command -v "${tool}")" "${BATS_TEST_TMPDIR}/bare/${tool}"
  done
  echo "${BATS_TEST_TMPDIR}/bare"
}

# Make a venv of a Python 3.12 with the pip that the Python bundles, and set VENV_PYTHON. pip reads only the folder
# wheels/, so the tests need no network. A computer without such a Python fails the test.
real_venv() {
  local base
  for base in python3.12 python3 python "$(uv python find 3.12 2> /dev/null || true)"; do
    [ -n "${base}" ] || continue
    [ "$("${base}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2> /dev/null)" = 3.12 ] || continue
    "${base}" -m venv --clear venv > /dev/null 2>&1 || continue
    VENV_PYTHON="${PWD}/venv/bin/python"
    export PIP_NO_INDEX=1 PIP_FIND_LINKS="${PWD}/wheels" PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_CONFIG_FILE=/dev/null
    unset PIP_CONSTRAINT PIP_REQUIRE_HASHES
    return
  done
  echo "no Python 3.12 here makes a venv with pip: run 'uv python install 3.12'"
  return 1
}

# Succeed when the pip of the venv is older than 26.1, the first pip that takes the pin of a named tool from a
# constraints file with digests.
pip_before_26_1() {
  "${VENV_PYTHON}" -c 'import pip, sys; sys.exit(tuple(map(int, pip.__version__.split(".")[:2])) >= (26, 1))'
}

# Write the wheels of five small stand-ins for the build tools into wheels/, and tools.txt, the requirements file
# that lists them with their digests. As the real tools do, `build` needs packaging and pyproject_hooks, and
# `wheel` needs packaging. The stand-in for `python -m build` writes its arguments into dist/. Without
# --no-isolation it first asks pip for the setuptools of the project, as the isolated build does.
write_tools() {
  "${VENV_PYTHON}" - <<'PY'
import hashlib, pathlib, zipfile

BUILD = '''import pathlib, subprocess, sys, tempfile
args = sys.argv[1:]
if "--no-isolation" not in args:
    env = tempfile.mkdtemp(dir=".")
    status = subprocess.call([sys.executable, "-m", "pip", "install", "--target", env, "setuptools>=77"])
    if status:
        sys.exit(status)
import setuptools
pathlib.Path("dist").mkdir()
pathlib.Path("dist/x-1-py3-none-any.whl").write_text(" ".join(args))
'''
NEEDS = {"build": ["packaging", "pyproject_hooks"], "wheel": ["packaging"], "packaging": [], "pyproject_hooks": [], "setuptools": []}
pathlib.Path("wheels").mkdir()
lines = []
for name, needs in NEEDS.items():
    info = f"{name}-99.dist-info"
    files = {
        f"{name}/__init__.py": "",
        f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: 99\n" + "".join(f"Requires-Dist: {need}\n" for need in needs),
        f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    if name == "build":
        files["build/__main__.py"] = BUILD
    files[f"{info}/RECORD"] = "".join(f"{path},,\n" for path in [*files, f"{info}/RECORD"])
    wheel = pathlib.Path("wheels", f"{name}-99-py3-none-any.whl")
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, text in files.items():
            archive.writestr(path, text)
    lines.append(f"{name}==99 --hash=sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}\n")
pathlib.Path("tools.txt").write_text("".join(lines))
PY
}

# Print how many of the stand-in tools the venv holds.
installed_tools() {
  "${VENV_PYTHON}" -m pip list --format freeze | grep -c '==99$' || true
}

@test "install_build_tools installs every tool of the file with the pip that Python 3.12 bundles" {
  real_venv
  write_tools
  run strict install_build_tools "${VENV_PYTHON}" tools.txt
  [ "$status" -eq 0 ]
  [ "$(installed_tools)" -eq 5 ]
}

@test "install_build_tools reads more than one file" {
  real_venv
  write_tools
  head -n 2 tools.txt > first.txt
  tail -n +3 tools.txt > second.txt
  run strict install_build_tools "${VENV_PYTHON}" first.txt second.txt
  [ "$status" -eq 0 ]
  [ "$(installed_tools)" -eq 5 ]
}

@test "install_build_tools refuses a tool with another digest" {
  real_venv
  write_tools
  sed "s/^setuptools==99 .*/setuptools==99 --hash=sha256:$(printf '%064d' 0)/" tools.txt > other.txt
  run strict install_build_tools "${VENV_PYTHON}" other.txt
  [ "$status" -eq 1 ]
  [[ "$output" == *"DO NOT MATCH THE HASHES"* ]]
  [ "$(installed_tools)" -eq 0 ]
}

@test "install_build_tools refuses a tool that needs one the file does not list" {
  real_venv
  write_tools
  grep -v '^packaging==' tools.txt > short.txt
  run strict install_build_tools "${VENV_PYTHON}" short.txt
  [ "$status" -eq 1 ]
  [[ "$output" == *"In --require-hashes mode, all requirements must have their versions pinned with ==. These do not:"*packaging* ]]
  [ "$(installed_tools)" -eq 0 ]
}

@test "install_build_tools refuses a file without digests" {
  real_venv
  write_tools
  sed 's/ --hash=.*//' tools.txt > plain.txt
  run strict install_build_tools "${VENV_PYTHON}" plain.txt
  [ "$status" -eq 1 ]
  [[ "$output" == *"Hashes are required in --require-hashes mode"* ]]
  [ "$(installed_tools)" -eq 0 ]
}

@test "install_build_tools fails without a file" {
  run strict install_build_tools python
  [ "$status" -eq 2 ]
  [[ "$output" == *"name a requirements file"* ]]
}

@test "install_build_tools refuses a file that pins no tool" {
  real_venv
  write_tools
  touch empty.txt
  printf '%s\n' '# build==1' '' '--require-hashes' > comments.txt
  local file
  for file in empty.txt comments.txt no-such-file.txt; do
    run strict install_build_tools "${VENV_PYTHON}" tools.txt "${file}"
    [ "$status" -eq 2 ]
    [[ "$output" == *"install_build_tools: ${file} pins no tool"* ]]
    [ "$(installed_tools)" -eq 0 ]
  done
}

@test "install_build_tools replaces a tool that the Python already holds" {
  real_venv
  write_tools
  "${VENV_PYTHON}" -m pip install setuptools
  echo "changed = True" > venv/lib/python3.12/site-packages/setuptools/__init__.py
  run strict install_build_tools "${VENV_PYTHON}" tools.txt
  [ "$status" -eq 0 ]
  [ "$(installed_tools)" -eq 5 ]
  [ ! -s venv/lib/python3.12/site-packages/setuptools/__init__.py ]
}

@test "install_build_tools checks the digest of a tool that the Python already holds" {
  real_venv
  write_tools
  "${VENV_PYTHON}" -m pip install setuptools
  sed "s/^setuptools==99 .*/setuptools==99 --hash=sha256:$(printf '%064d' 0)/" tools.txt > other.txt
  run strict install_build_tools "${VENV_PYTHON}" other.txt
  [ "$status" -eq 1 ]
  [[ "$output" == *"DO NOT MATCH THE HASHES"* ]]
  [ "$(installed_tools)" -eq 1 ]
}

@test "build_wheel_with_tools builds without isolation, also where the shell holds a constraints file" {
  real_venv
  write_tools
  strict install_build_tools "${VENV_PYTHON}" tools.txt
  PIP_CONSTRAINT=tools.txt run strict build_wheel_with_tools "${VENV_PYTHON}" -C--build-option=--build-number=1.cpu
  [ "$status" -eq 0 ]
  [ "$(cat dist/x-1-py3-none-any.whl)" = "--wheel --no-isolation -C--build-option=--build-number=1.cpu" ]
}

@test "only a pip of 26.1 or newer installs named tools that a constraints file pins by digest" {
  real_venv
  write_tools
  PIP_CONSTRAINT=tools.txt run "${VENV_PYTHON}" -m pip install build wheel
  if pip_before_26_1; then
    [ "$status" -eq 1 ]
    [[ "$output" == *"In --require-hashes mode, all requirements must have their versions pinned with ==. These do not:"*build* ]]
    [ "$(installed_tools)" -eq 0 ]
  else
    [ "$status" -eq 0 ]
    [ "$(installed_tools)" -eq 4 ]
  fi
  # The call of install_build_tools works with each pip.
  real_venv
  run strict install_build_tools "${VENV_PYTHON}" tools.txt
  [ "$status" -eq 0 ]
  [ "$(installed_tools)" -eq 5 ]
}

@test "only a pip of 26.1 or newer does an isolated build with a constraints file that holds digests" {
  real_venv
  write_tools
  strict install_build_tools "${VENV_PYTHON}" tools.txt
  PIP_CONSTRAINT=tools.txt run "${VENV_PYTHON}" -m build --wheel
  if pip_before_26_1; then
    [ "$status" -eq 1 ]
    [[ "$output" == *"In --require-hashes mode, all requirements must have their versions pinned with ==. These do not:"*setuptools* ]]
    [ ! -e dist ]
  else
    [ "$status" -eq 0 ]
    [ "$(cat dist/x-1-py3-none-any.whl)" = "--wheel" ]
  fi
}

# Print each requirement of a requirements file that has no exact version or no sha256. A backslash joins two lines.
loose_requirements() {  # file
  awk '{ sub(/\r$/, ""); line = line $0 } !sub(/\\$/, "", line) { print line; line = "" } END { print line }' "$1" > joined.txt || return 1
  grep -vE -e '^[[:space:]]*(#|$)' \
    -e '^[A-Za-z0-9][A-Za-z0-9._-]*==[0-9][0-9A-Za-z.]*([[:space:]]+--hash=sha256:[0-9a-f]{64})+[[:space:]]*$' joined.txt || [ "$?" -eq 1 ]
}

@test "each requirements file of the pack tools gives every tool one version and a sha256" {
  local file loose
  for file in "${REPO_ROOT}"/tools/wheel-build/*.constraints; do
    loose="$(loose_requirements "${file}")"
    [ -z "${loose}" ] || { echo "${file}: ${loose}"; return 1; }
  done
}

@test "a requirement without an exact version or a sha256 is loose" {
  local digest line
  digest="--hash=sha256:$(printf '%064d' 0)"
  printf '%s\n' '# tools' '' "a==1.2 ${digest}" 'b_c==3 \' "    ${digest} \\" "    ${digest}" > exact.txt
  [ -z "$(loose_requirements exact.txt)" ]
  for line in 'd==1' "d>=1 ${digest}" "d ${digest}" "d==1.* ${digest}" "d==1 ; python_version < '3' ${digest}" "-r other.txt" \
    "d==1 ${digest}0" "d==1 --hash=sha256:0"; do
    printf '%s\n' "a==1.2 ${digest}" "${line}" "b==2 ${digest}" > one.txt
    [ "$(loose_requirements one.txt)" = "${line}" ] || { echo "not loose: ${line}"; return 1; }
  done
}

@test "fetch_release_asset downloads the named asset of the tag" {
  run strict fetch_release_asset owner/repo v1 a.tar.gz upstream
  [ "$status" -eq 0 ]
  [ "$(cat upstream/a.tar.gz)" = archive ]
  [ ! -e upstream/b.tar.gz ]
  grep -qx 'release download v1 -R owner/repo --pattern a.tar.gz --dir upstream' "${GH_LOG}"
  grep -q '^release view v1 -R owner/repo --json assets' "${GH_LOG}"
}

@test "fetch_release_asset fails when the digest differs" {
  GH_DIGEST=sha256:0000 run strict fetch_release_asset owner/repo v1 a.tar.gz upstream
  [ "$status" -eq 1 ]
  [[ "$output" == *"digest mismatch a.tar.gz: sha256:0000 sha256:"* ]]
}

@test "fetch_release_asset fails when GitHub lists no digest" {
  GH_DIGEST=null run strict fetch_release_asset owner/repo v1 a.tar.gz upstream
  [ "$status" -eq 1 ]
}

@test "fetch_release_asset fails when the release lacks the asset" {
  run strict fetch_release_asset owner/repo v1 c.tar.gz upstream
  [ "$status" -eq 1 ]
}

@test "fetch_release_asset fails when gh fails after it wrote the file" {
  GH_DOWNLOAD_STATUS=1 run strict fetch_release_asset owner/repo v1 a.tar.gz upstream
  [ "$status" -eq 1 ]
}

@test "fetch_release_asset fails in a condition too, where the shell ignores set -e" {
  GH_DIGEST=sha256:0000 run strict eval 'if fetch_release_asset owner/repo v1 a.tar.gz upstream; then echo passed; else echo refused; fi'
  [[ "$output" == *refused* ]]
  [[ "$output" != *passed* ]]
}

@test "sha256_line writes the digest and the file name without its folder" {
  run strict sha256_line assets/a.tar.gz
  [ "$status" -eq 0 ]
  [[ "$output" =~ ^[0-9a-f]{64}\ [\ *]a\.tar\.gz$ ]]
}

@test "sha256_line uses shasum where the computer has no sha256sum" {
  local line bare
  line="$(strict sha256_line assets/a.tar.gz)"
  bare="$(without_sha256sum)"
  PATH="${bare}" run strict sha256_line assets/a.tar.gz
  [ "$status" -eq 0 ]
  [ "${output%% *}" = "${line%% *}" ]
  [ "${output##*[ *]}" = a.tar.gz ]
}

@test "merge_terms keeps the mistral.rs order: MIT, the crates, then the other terms" {
  run strict eval '{ echo MIT; printf "0BSD\nApache-2.0\nBSD-3-Clause\nMIT\nZlib\n"; printf "Apache-2.0\nBSD-3-Clause\nLicenseRef-Microsoft-VC-Runtime\n"; } | merge_terms'
  [ "$status" -eq 0 ]
  [ "$output" = "MIT AND 0BSD AND Apache-2.0 AND BSD-3-Clause AND Zlib AND LicenseRef-Microsoft-VC-Runtime" ]
}

@test "merge_terms keeps the order of the cell before the sorted terms, as the llama.cpp and ik packs do" {
  run strict eval '{ printf "MIT\nLicenseRef-NVIDIA-CUDA-EULA\n"; printf "Unlicense\nMIT\nBSD-3-Clause\n" | sort -u; } | merge_terms'
  [ "$status" -eq 0 ]
  [ "$output" = "MIT AND LicenseRef-NVIDIA-CUDA-EULA AND BSD-3-Clause AND Unlicense" ]
}

@test "merge_terms drops carriage returns and empty lines" {
  run strict eval 'printf "MIT\r\n\r\n\nISC\r\nMIT\n" | merge_terms'
  [ "$status" -eq 0 ]
  [ "$output" = "MIT AND ISC" ]
}

@test "merge_terms keeps a term that holds a space as one term" {
  run strict eval 'printf "Apache-2.0 WITH LLVM-exception\nApache-2.0\n" | merge_terms'
  [ "$output" = "Apache-2.0 WITH LLVM-exception AND Apache-2.0" ]
}

@test "merge_terms fails without a term" {
  run strict eval 'printf "\n" | merge_terms'
  [ "$status" -ne 0 ]
}

@test "collect_wheel copies the wheel and writes its sha256 file" {
  mkdir dist
  echo wheel > dist/x-1-py3-none-any.whl
  run strict collect_wheel dist out
  [ "$status" -eq 0 ]
  [ "$(cat out/x-1-py3-none-any.whl)" = wheel ]
  (cd out && sha256sum -c x-1-py3-none-any.whl.sha256)
}

@test "collect_wheel fails on two wheels" {
  mkdir dist
  echo a > dist/a.whl
  echo b > dist/b.whl
  run strict collect_wheel dist out
  [ "$status" -eq 1 ]
  [[ "$output" == *"expected one wheel in out, found 2"* ]]
}

@test "collect_wheel fails on a wheel of an earlier build in the output folder" {
  mkdir dist out
  echo a > dist/a.whl
  echo b > out/b.whl
  run strict collect_wheel dist out
  [ "$status" -eq 1 ]
}

@test "collect_wheel fails without a wheel" {
  mkdir dist
  run strict collect_wheel dist out
  [ "$status" -ne 0 ]
}

@test "host_os reads the runner, then the system" {
  cp "${BATS_TEST_DIRNAME}/stubs/uname" "${BIN}/uname"
  [ "$(RUNNER_OS=Windows strict host_os)" = Windows ]
  unset RUNNER_OS
  local pair
  for pair in Linux:Linux Darwin:macOS MINGW64_NT-10.0:Windows MSYS_NT-10.0:Windows; do
    printf '#!/bin/sh\necho %s\n' "${pair%%:*}" > "${BIN}/uname"
    chmod +x "${BIN}/uname"
    [ "$(strict host_os)" = "${pair#*:}" ]
  done
  printf '#!/bin/sh\necho Plan9\n' > "${BIN}/uname"
  run strict host_os
  [ "$status" -eq 1 ]
}

@test "python_312 takes the first interpreter that is 3.12" {
  printf '#!/bin/sh\necho 3.11\n' > "${BIN}/python"
  printf '#!/bin/sh\necho 3.12\n' > "${BIN}/python3.12"
  chmod +x "${BIN}/python" "${BIN}/python3.12"
  [ "$(strict python_312)" = python3.12 ]
  printf '#!/bin/sh\necho 3.12\n' > "${BIN}/python"
  [ "$(strict python_312)" = python ]
}

@test "python_312 passes over an interpreter that fails" {
  printf '#!/bin/sh\nexit 127\n' > "${BIN}/python"
  printf '#!/bin/sh\necho 3.12\n' > "${BIN}/python3.12"
  chmod +x "${BIN}/python" "${BIN}/python3.12"
  [ "$(strict python_312)" = python3.12 ]
}

@test "python_312 fails without a 3.12" {
  local name
  for name in python python3.12 python3; do
    printf '#!/bin/sh\necho 3.13\n' > "${BIN}/${name}"
    chmod +x "${BIN}/${name}"
  done
  run strict python_312
  [ "$status" -eq 1 ]
  [[ "$output" == *"Python 3.12 is not on PATH"* ]]
}

# assert_libraries_resolve: the check that the manylinux build scripts run on llama-server.
ldd_stub() {  # text status
  printf '#!/bin/sh\necho "%s"\nexit %s\n' "$1" "$2" > "${BIN}/ldd"
  chmod +x "${BIN}/ldd"
}

real_ldd() {
  command -v ldd > /dev/null || skip "ldd is a Linux tool"
}

@test "assert_libraries_resolve refuses a text file that ldd cannot read" {
  real_ldd
  echo "not a program" > "${BATS_TEST_TMPDIR}/text"
  chmod +x "${BATS_TEST_TMPDIR}/text"
  run strict assert_libraries_resolve "${BATS_TEST_TMPDIR}/text"
  [ "$status" -ne 0 ]
  [[ "$output" == *"ldd failed on ${BATS_TEST_TMPDIR}/text"* ]]
}

@test "assert_libraries_resolve refuses an ELF file of a machine that ldd cannot read" {
  real_ldd
  # An ELF header with a machine number that no host runs.
  printf '\177ELF\002\001\001\000\000\000\000\000\000\000\000\000\002\000\377\377\001\000\000\000' > "${BATS_TEST_TMPDIR}/alien"
  head -c 64 /dev/zero >> "${BATS_TEST_TMPDIR}/alien"
  chmod +x "${BATS_TEST_TMPDIR}/alien"
  run strict assert_libraries_resolve "${BATS_TEST_TMPDIR}/alien"
  [ "$status" -ne 0 ]
  [[ "$output" == *"ldd failed on ${BATS_TEST_TMPDIR}/alien"* ]]
}

@test "assert_libraries_resolve refuses a path that is not there" {
  real_ldd
  run strict assert_libraries_resolve "${BATS_TEST_TMPDIR}/missing"
  [ "$status" -ne 0 ]
  [[ "$output" == *"ldd failed on ${BATS_TEST_TMPDIR}/missing"* ]]
}

@test "assert_libraries_resolve passes a dynamic binary whose libraries resolve" {
  real_ldd
  run strict assert_libraries_resolve "$(command -v ls)"
  [ "$status" -eq 0 ]
}

@test "assert_libraries_resolve refuses a library that is not found" {
  ldd_stub "libggml.so => not found" 0
  run strict assert_libraries_resolve /x/llama-server
  [ "$status" -ne 0 ]
  [[ "$output" == *"llama-server has an unresolved library"* ]]
  [[ "$output" == *"libggml.so => not found"* ]]
}

@test "assert_libraries_resolve passes a missing library that the host supplies" {
  ldd_stub "libcuda.so.1 => not found" 0
  run strict assert_libraries_resolve /x/llama-server 'libcuda\.so\.1'
  [ "$status" -eq 0 ]
}

@test "assert_libraries_resolve refuses another missing library beside a host library" {
  ldd_stub "libcuda.so.1 => not found
libggml.so => not found" 0
  run strict assert_libraries_resolve /x/llama-server 'libcuda\.so\.1'
  [ "$status" -ne 0 ]
  [[ "$output" == *"libggml.so => not found"* ]]
}
