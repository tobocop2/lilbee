#!/usr/bin/env bats
# site/install.sh: the build it picks for every machine case, and what it does with a download.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/site/install.sh"
  CASES="${REPO_ROOT}/tests/installer/cases.tsv"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  export FIXTURE
  mkdir -p "${FIXTURE}/bin" "${FIXTURE}/release" "${BATS_TEST_TMPDIR}/home"
  : > "${FIXTURE}/curl.log"
  cp "${BATS_TEST_DIRNAME}/stubs/curl-release" "${FIXTURE}/bin/curl"
  for stub in uname getconf ldd sw_vers; do cp "${BATS_TEST_DIRNAME}/stubs/${stub}" "${FIXTURE}/bin/${stub}"; done
  export HOME="${BATS_TEST_TMPDIR}/home"
  export SHELL=/bin/zsh
  export LILBEE_INSTALL_DIR="${HOME}/bin"
  export LILBEE_VARIANT=default
  export UNAME_S=Linux UNAME_M=x86_64
  unset LILBEE_RELEASE LILBEE_NO_MODIFY_PATH GITHUB_ACTIONS GITHUB_PATH XDG_BIN_HOME
  PATH="${FIXTURE}/bin:${PATH}"
}

hash_of() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# Publish asset $1 with body $2 and a SHA256SUMS line that matches it, or $3 as the hash.
publish() {
  printf '%s' "$2" > "${FIXTURE}/release/$1"
  printf '%s  %s\n' "${3:-$(hash_of "${FIXTURE}/release/$1")}" "$1" >> "${FIXTURE}/release/SHA256SUMS"
}

install_lilbee() { run sh "${SCRIPT}"; }

# A PATH folder holding only what install.sh needs, plus the tools named in the arguments.
tool_dir() {
  tools="${BATS_TEST_TMPDIR}/tools"
  mkdir -p "${tools}"
  for t in sh bash env uname getconf ldd awk grep sed tr head sort paste mktemp df mv cp chmod rm cat dirname mkdir "$@"; do
    if command -v "${t}" >/dev/null; then ln -sf "$(command -v "${t}")" "${tools}/${t}"; fi
  done
}

@test "every machine case gets the build the case table names" {
  total=0
  matched=0
  while read -r id os arch avx2 cuda kfd gfx userland variant expected; do
    total=$((total + 1))
    [ "${variant}" = "-" ] && variant=""
    got=$(sh "${REPO_ROOT}/tests/installer/select_asset.sh" "${os}" "${arch}" "${avx2}" "${cuda}" "${kfd}" "${gfx}" "${userland}" "${variant}" 2>/dev/null) || got=ERROR
    if [ "${got}" = "${expected}" ]; then matched=$((matched + 1)); else echo "${id}: expected ${expected}, got ${got}" >&3; fi
  done < <(awk -F'\t' 'NR > 1 { v = $9; if (v == "") v = "-"; print $1, $2, $3, $4, $5, $6, $7, $8, v, $10 }' "${CASES}")
  echo "matched ${matched} of ${total}" >&3
  [ "${total}" -gt 100 ]
  [ "${matched}" -eq "${total}" ]
}

@test "the macOS and glibc floors match the release cell definitions" {
  cells="${REPO_ROOT}/.github/release-cells.yml"
  arm=$(awk '/asset_name: lilbee-macos-arm64$/ { f = 1 } f && /macos_floor/ { gsub(/[" ]|macos_floor:/, ""); print; exit }' "${cells}")
  intel=$(awk '/asset_name: lilbee-macos-x86_64$/ { f = 1 } f && /macos_floor/ { gsub(/[" ]|macos_floor:/, ""); print; exit }' "${cells}")
  glibc=$(grep -o 'manylinux_2_[0-9]*' "${cells}" | head -n 1 | sed 's/manylinux_2_//')
  [ -n "${arm}" ] && [ -n "${intel}" ] && [ -n "${glibc}" ]
  grep -qx "MACOS_FLOOR_ARM64=\"${arm}\"" "${SCRIPT}"
  grep -qx "MACOS_FLOOR_X86_64=\"${intel}\"" "${SCRIPT}"
  grep -qx "GLIBC_FLOOR=\"2.${glibc}\"" "${SCRIPT}"
}

@test "a matching download is installed and runnable" {
  publish lilbee-linux-x86_64 $'#!/bin/sh\necho lilbee-fake\n'
  install_lilbee
  [ "$status" -eq 0 ]
  [[ "$output" == *"The SHA-256 matches SHA256SUMS"* ]]
  [ -x "${LILBEE_INSTALL_DIR}/lilbee" ]
  [ "$("${LILBEE_INSTALL_DIR}/lilbee")" = "lilbee-fake" ]
}

@test "a download that does not match its SHA256SUMS line installs nothing" {
  publish lilbee-linux-x86_64 "tampered" "$(printf 'x' | shasum -a 256 | cut -d' ' -f1)"
  install_lilbee
  [ "$status" -ne 0 ]
  [[ "$output" == *"does not match SHA256SUMS. Nothing was installed"* ]]
  [ ! -e "${LILBEE_INSTALL_DIR}/lilbee" ]
  [ -z "$(find "${LILBEE_INSTALL_DIR}" -name '.lilbee.*')" ]
}

@test "a mismatch leaves the installed file as it was" {
  mkdir -p "${LILBEE_INSTALL_DIR}"
  printf 'old' > "${LILBEE_INSTALL_DIR}/lilbee"
  publish lilbee-linux-x86_64 "tampered" "$(printf 'x' | shasum -a 256 | cut -d' ' -f1)"
  install_lilbee
  [ "$status" -ne 0 ]
  [ "$(cat "${LILBEE_INSTALL_DIR}/lilbee")" = "old" ]
}

@test "a release without SHA256SUMS is refused and the message names npm" {
  printf 'body' > "${FIXTURE}/release/lilbee-linux-x86_64"
  install_lilbee
  [ "$status" -ne 0 ]
  [[ "$output" == *"no SHA256SUMS"* ]]
  [[ "$output" == *"npm install -g lilbee"* ]]
  [ ! -e "${LILBEE_INSTALL_DIR}/lilbee" ]
}

@test "a SHA256SUMS without a line for the asset is refused" {
  publish lilbee-macos-arm64 "other"
  install_lilbee
  [ "$status" -ne 0 ]
  [[ "$output" == *"no lilbee build for this machine"* ]]
  [ ! -e "${LILBEE_INSTALL_DIR}/lilbee" ]
}

@test "an installed file that equals the expected line is not downloaded again" {
  publish lilbee-linux-x86_64 "current"
  mkdir -p "${LILBEE_INSTALL_DIR}"
  printf 'current' > "${LILBEE_INSTALL_DIR}/lilbee"
  install_lilbee
  [ "$status" -eq 0 ]
  [[ "$output" == *"up to date"* ]]
  [ "$(grep -c 'lilbee-linux-x86_64$' "${FIXTURE}/curl.log")" -eq 0 ]
}

@test "an older installed file is replaced" {
  publish lilbee-linux-x86_64 "new"
  mkdir -p "${LILBEE_INSTALL_DIR}"
  printf 'old' > "${LILBEE_INSTALL_DIR}/lilbee"
  install_lilbee
  [ "$status" -eq 0 ]
  [ "$(cat "${LILBEE_INSTALL_DIR}/lilbee")" = "new" ]
}

@test "LILBEE_RELEASE pins the release the files come from" {
  publish lilbee-linux-x86_64 "pinned"
  LILBEE_RELEASE=v1.2.3 install_lilbee
  [ "$status" -eq 0 ]
  grep -q 'releases/download/v1.2.3/SHA256SUMS' "${FIXTURE}/curl.log"
  grep -q 'releases/download/v1.2.3/lilbee-linux-x86_64' "${FIXTURE}/curl.log"
}

@test "without a pin the files come from the latest release" {
  publish lilbee-linux-x86_64 "latest"
  install_lilbee
  grep -q 'releases/latest/download/SHA256SUMS' "${FIXTURE}/curl.log"
}

@test "an unknown LILBEE_VARIANT is refused with the npm launcher's text" {
  publish lilbee-linux-x86_64 "body"
  LILBEE_VARIANT=cu999 install_lilbee
  [ "$status" -ne 0 ]
  [[ "$output" == *'Unknown LILBEE_VARIANT "cu999". Valid: default, cu121, cu124, cu125, rocm, compat, compat-cu124, compat-rocm, or unset.'* ]]
}

@test "a GPU variant that the release does not ship falls back to the plain build" {
  publish lilbee-linux-x86_64 "plain"
  LILBEE_VARIANT=cu125 install_lilbee
  [ "$status" -eq 0 ]
  [ "$(cat "${LILBEE_INSTALL_DIR}/lilbee")" = "plain" ]
}

@test "no curl and no wget is a clean error" {
  publish lilbee-linux-x86_64 "body"
  tool_dir sha256sum shasum
  run env PATH="${tools}" /bin/sh "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"curl or wget is required"* ]]
}

@test "no sha256 tool is a clean error" {
  publish lilbee-linux-x86_64 "body"
  tool_dir curl
  run env PATH="${tools}" /bin/sh "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"sha256sum or shasum is required"* ]]
}

@test "too little free space stops before the download" {
  publish lilbee-linux-x86_64 "body"
  cp "${BATS_TEST_DIRNAME}/stubs/df-low" "${FIXTURE}/bin/df"
  install_lilbee
  [ "$status" -ne 0 ]
  [[ "$output" == *"not enough free space"* ]]
  ! grep -q 'lilbee-linux-x86_64$' "${FIXTURE}/curl.log"
}

@test "a glibc below the floor is refused" {
  publish lilbee-linux-x86_64 "body"
  GLIBC=2.31 install_lilbee
  [ "$status" -ne 0 ]
  [[ "$output" == *"needs glibc 2.34 or newer. This system has 2.31"* ]]
}

@test "glibc at the floor is accepted" {
  publish lilbee-linux-x86_64 "body"
  GLIBC=2.34 install_lilbee
  [ "$status" -eq 0 ]
}

@test "a musl system is refused" {
  publish lilbee-linux-x86_64 "body"
  LIBC=musl install_lilbee
  [ "$status" -ne 0 ]
  [[ "$output" == *"musl"* ]]
}

@test "Linux arm64 is refused before any download" {
  UNAME_M=aarch64 install_lilbee
  [ "$status" -ne 0 ]
  [[ "$output" == *"no standalone lilbee build for Linux arm64"* ]]
  [ ! -s "${FIXTURE}/curl.log" ]
}

@test "Windows is pointed at install.ps1" {
  UNAME_S=MINGW64_NT-10.0 install_lilbee
  [ "$status" -ne 0 ]
  [[ "$output" == *"install.ps1"* ]]
}

@test "macOS below the floor of the arm64 build is refused" {
  publish lilbee-macos-arm64 "body"
  UNAME_S=Darwin UNAME_M=arm64 MACOS=13.6 install_lilbee
  [ "$status" -ne 0 ]
  [[ "$output" == *"needs macOS 14.0 or newer. This Mac runs 13.6"* ]]
}

@test "macOS at the floor of the arm64 build installs" {
  publish lilbee-macos-arm64 "body"
  UNAME_S=Darwin UNAME_M=arm64 MACOS=14.0 install_lilbee
  [ "$status" -eq 0 ]
}

@test "an Intel Mac needs macOS 12 and gets the plain Intel build" {
  publish lilbee-macos-x86_64 "intel"
  UNAME_S=Darwin MACOS=12.0 install_lilbee
  [ "$status" -eq 0 ]
  [ "$(cat "${LILBEE_INSTALL_DIR}/lilbee")" = "intel" ]
}

@test "a folder that is not on PATH is added to the shell startup file once" {
  publish lilbee-linux-x86_64 "body"
  install_lilbee
  [ "$status" -eq 0 ]
  install_lilbee
  [ "$(grep -c "export PATH=\"${LILBEE_INSTALL_DIR}:" "${HOME}/.zshrc")" -eq 1 ]
}

@test "LILBEE_NO_MODIFY_PATH leaves the shell startup files alone and prints the line" {
  publish lilbee-linux-x86_64 "body"
  LILBEE_NO_MODIFY_PATH=1 install_lilbee
  [ "$status" -eq 0 ]
  [ ! -e "${HOME}/.zshrc" ]
  [[ "$output" == *"export PATH=\"${LILBEE_INSTALL_DIR}:"* ]]
}

@test "a folder already on PATH needs no startup file change" {
  publish lilbee-linux-x86_64 "body"
  PATH="${LILBEE_INSTALL_DIR}:${PATH}" install_lilbee
  [ "$status" -eq 0 ]
  [ ! -e "${HOME}/.zshrc" ]
}

@test "inside GitHub Actions the folder is appended to GITHUB_PATH" {
  publish lilbee-linux-x86_64 "body"
  GITHUB_ACTIONS=true GITHUB_PATH="${BATS_TEST_TMPDIR}/gh_path" LILBEE_NO_MODIFY_PATH=1 install_lilbee
  [ "$status" -eq 0 ]
  [ "$(cat "${BATS_TEST_TMPDIR}/gh_path")" = "${LILBEE_INSTALL_DIR}" ]
}

@test "another lilbee earlier on PATH is named in a warning" {
  publish lilbee-linux-x86_64 "body"
  mkdir -p "${BATS_TEST_TMPDIR}/other"
  printf '#!/bin/sh\n' > "${BATS_TEST_TMPDIR}/other/lilbee"
  chmod +x "${BATS_TEST_TMPDIR}/other/lilbee"
  PATH="${BATS_TEST_TMPDIR}/other:${PATH}" install_lilbee
  [ "$status" -eq 0 ]
  [[ "$output" == *"another lilbee comes first on PATH: ${BATS_TEST_TMPDIR}/other/lilbee"* ]]
}

@test "the final line of the script is the only top-level call" {
  [ "$(tail -n 1 "${SCRIPT}")" = '[ -n "${LILBEE_INSTALL_LIB:-}" ] || main "$@"' ]
  [ "$(grep -c '^main ' "${SCRIPT}")" -eq 0 ]
}

@test "bash on Linux gets the line in .bashrc, once" {
  publish lilbee-linux-x86_64 "body"
  SHELL=/bin/bash install_lilbee
  [ "$status" -eq 0 ]
  SHELL=/bin/bash install_lilbee
  [ "$(grep -c "export PATH=\"${LILBEE_INSTALL_DIR}:" "${HOME}/.bashrc")" -eq 1 ]
  [ ! -e "${HOME}/.profile" ]
}

@test "bash on a Mac gets the line in .bash_profile" {
  publish lilbee-macos-arm64 "body"
  UNAME_S=Darwin UNAME_M=arm64 SHELL=/bin/bash install_lilbee
  [ "$status" -eq 0 ]
  grep -q "export PATH=\"${LILBEE_INSTALL_DIR}:" "${HOME}/.bash_profile"
  [ ! -e "${HOME}/.bashrc" ]
}

@test "sh gets the line in .profile" {
  publish lilbee-linux-x86_64 "body"
  SHELL=/bin/sh install_lilbee
  [ "$status" -eq 0 ]
  grep -q "export PATH=\"${LILBEE_INSTALL_DIR}:" "${HOME}/.profile"
}

@test "zsh honours ZDOTDIR" {
  publish lilbee-linux-x86_64 "body"
  mkdir -p "${BATS_TEST_TMPDIR}/zdot"
  ZDOTDIR="${BATS_TEST_TMPDIR}/zdot" install_lilbee
  [ "$status" -eq 0 ]
  grep -q "export PATH=\"${LILBEE_INSTALL_DIR}:" "${BATS_TEST_TMPDIR}/zdot/.zshrc"
  [ ! -e "${HOME}/.zshrc" ]
}

@test "fish gets a conf.d file with fish syntax, once, and no .profile" {
  publish lilbee-linux-x86_64 "body"
  SHELL=/usr/bin/fish install_lilbee
  [ "$status" -eq 0 ]
  SHELL=/usr/bin/fish install_lilbee
  conf="${HOME}/.config/fish/conf.d/lilbee.fish"
  [ "$(grep -c "fish_add_path --global \"${LILBEE_INSTALL_DIR}\"" "${conf}")" -eq 1 ]
  ! grep -q 'export PATH' "${conf}"
  [ ! -e "${HOME}/.profile" ]
  [[ "$output" == *"fish_add_path --global"* ]]
}

@test "fish honours XDG_CONFIG_HOME" {
  publish lilbee-linux-x86_64 "body"
  XDG_CONFIG_HOME="${BATS_TEST_TMPDIR}/xdg" SHELL=/usr/bin/fish install_lilbee
  [ "$status" -eq 0 ]
  [ -s "${BATS_TEST_TMPDIR}/xdg/fish/conf.d/lilbee.fish" ]
  [ ! -e "${HOME}/.config" ]
}

@test "an interrupt during the download removes the partial file and the work folder" {
  run sh "${BATS_TEST_DIRNAME}/stubs/int-trappable"
  [ "$status" -eq 0 ] || skip "SIGINT is ignored in this run, so no script can trap it"
  publish lilbee-linux-x86_64 "body"
  cp "${BATS_TEST_DIRNAME}/stubs/curl-interrupted" "${FIXTURE}/bin/curl"
  export TMPDIR="${BATS_TEST_TMPDIR}/tmp"
  mkdir -p "${TMPDIR}"
  install_lilbee
  [ "$status" -eq 130 ]
  [ -z "$(find "${LILBEE_INSTALL_DIR}" -name '.lilbee.*')" ]
  [ -z "$(find "${TMPDIR}" -mindepth 1)" ]
  [ ! -e "${LILBEE_INSTALL_DIR}/lilbee" ]
}

@test "a terminate signal during the download ends the install with status 143 and installs nothing" {
  publish lilbee-linux-x86_64 "body"
  cp "${BATS_TEST_DIRNAME}/stubs/curl-interrupted" "${FIXTURE}/bin/curl"
  SIGNAL=TERM install_lilbee
  [ "$status" -eq 143 ]
  [ -z "$(find "${LILBEE_INSTALL_DIR}" -name '.lilbee.*')" ]
  [ ! -e "${LILBEE_INSTALL_DIR}/lilbee" ]
}
