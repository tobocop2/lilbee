#!/bin/sh
# lilbee installer for macOS and Linux: picks the build for this machine,
# checks it against the release's SHA256SUMS, and installs one file.
#   curl -fsSL https://lilbee.sh/install.sh | sh
# Settings: LILBEE_INSTALL_DIR, LILBEE_RELEASE, LILBEE_VARIANT, LILBEE_NO_MODIFY_PATH.
set -eu

REPO="tobocop2/lilbee"
RELEASES_URL="https://github.com/${REPO}/releases"
MACOS_FLOOR_ARM64="14.0"
MACOS_FLOOR_X86_64="12.0"
GLIBC_FLOOR="2.34"
MIN_FREE_KB=2097152
# Tests point this at a fake root holding /proc, /sys, /dev and /opt.
FS_ROOT="${LILBEE_FS_ROOT:-}"
VARIANTS="default cu121 cu124 cu125 rocm compat compat-cu124 compat-rocm"
NPM_HINT="Install with npm (npm install -g lilbee) or download a binary from ${RELEASES_URL}."
OTHER_ROUTES="Use the Docker image or pip: https://github.com/${REPO}#install"

work=""
partial=""

say() { printf 'lilbee: %s\n' "$*"; }
warn() { printf 'lilbee: warning: %s\n' "$*" >&2; }
die() { printf 'lilbee: error: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

cleanup() {
  [ -z "${partial}" ] || rm -f "${partial}"
  [ -z "${work}" ] || rm -rf "${work}"
}

# True when dotted version $1 is at least $2.
version_ge() {
  vg_a="$1"
  vg_b="$2"
  while [ -n "${vg_a}${vg_b}" ]; do
    vg_ah="${vg_a%%.*}"
    vg_bh="${vg_b%%.*}"
    vg_ah="${vg_ah:-0}"
    vg_bh="${vg_bh:-0}"
    [ "${vg_ah}" -gt "${vg_bh}" ] && return 0
    [ "${vg_ah}" -lt "${vg_bh}" ] && return 1
    case "${vg_a}" in *.*) vg_a="${vg_a#*.}" ;; *) vg_a="" ;; esac
    case "${vg_b}" in *.*) vg_b="${vg_b#*.}" ;; *) vg_b="" ;; esac
  done
  return 0
}

# The CUDA build a driver ceiling such as 12.4 can run, or nothing.
cuda_build() {
  [ "$1" != "-" ] || return 0
  cb_ceiling=$(( ${1%%.*} * 100 + ${1#*.} ))
  if [ "${cb_ceiling}" -ge 1205 ]; then echo cu125
  elif [ "${cb_ceiling}" -ge 1204 ]; then echo cu124
  elif [ "${cb_ceiling}" -ge 1201 ]; then echo cu121
  fi
}

# The variant the hardware calls for: os arch avx2 cuda kfd gfx rocm_userland.
host_variant() {
  hv_os="$1" hv_arch="$2" hv_avx2="$3" hv_kfd="$5" hv_gfx="$6" hv_userland="$7"
  hv_build=$(cuda_build "$4")
  hv_compat=no
  [ "${hv_arch}" = x64 ] && [ "${hv_avx2}" = no ] && hv_compat=yes
  hv_rocm=no
  if [ "${hv_os}" = linux ] && [ -z "${hv_build}" ] && [ "${hv_kfd}" = yes ] \
    && { [ "${hv_gfx}" != "-" ] || [ "${hv_userland}" = yes ]; }; then
    hv_rocm=yes
  fi
  case "${hv_os}" in
    darwin)
      [ "${hv_compat}" = yes ] && echo compat || echo default
      ;;
    win32)
      if [ "${hv_compat}" = yes ]; then
        case "${hv_build}" in cu124 | cu125) echo compat-cu124 ;; *) echo compat ;; esac
      elif [ "${hv_build}" = cu121 ]; then echo default
      else echo "${hv_build:-default}"
      fi
      ;;
    *)
      if [ "${hv_compat}" = yes ]; then
        case "${hv_build}" in
          cu124 | cu125) echo compat-cu124 ;;
          *) [ "${hv_rocm}" = yes ] && echo compat-rocm || echo compat ;;
        esac
      elif [ -n "${hv_build}" ]; then echo "${hv_build}"
      else [ "${hv_rocm}" = yes ] && echo rocm || echo default
      fi
      ;;
  esac
}

# The release asset name for a host and variant; fails with the reason.
asset_name() {
  an_variant="$3"
  [ "${an_variant}" != default ] || an_variant=""
  case " ${VARIANTS} " in
    *" ${an_variant:-default} "*) ;;
    *) echo "Unknown LILBEE_VARIANT \"$3\". Valid: default, cu121, cu124, cu125, rocm, compat, compat-cu124, compat-rocm, or unset." >&2; return 1 ;;
  esac
  an_prefix=lilbee
  an_gpu="${an_variant}"
  case "${an_variant}" in
    compat | compat-*) an_prefix=lilbee-compat; an_gpu="${an_variant#compat}"; an_gpu="${an_gpu#-}" ;;
  esac
  case "$1/$2" in
    darwin/arm64)
      [ -z "${an_variant}" ] || { echo "macOS arm64 ships a single build; unset LILBEE_VARIANT." >&2; return 1; }
      echo lilbee-macos-arm64
      ;;
    darwin/x64)
      [ -z "${an_gpu}" ] || { echo "macOS has no GPU-variant builds; unset LILBEE_VARIANT." >&2; return 1; }
      echo "${an_prefix}-macos-x86_64"
      ;;
    linux/x64) echo "${an_prefix}-linux-x86_64${an_gpu:+-${an_gpu}}" ;;
    win32/x64)
      if [ "${an_prefix}" = lilbee-compat ] && [ -n "${an_gpu}" ] && [ "${an_variant}" != compat-cu124 ]; then
        echo "Windows compat build has only the compat-cu124 GPU variant." >&2
        return 1
      fi
      echo "${an_prefix}-windows-x86_64${an_gpu:+-${an_gpu}}.exe"
      ;;
    *) echo "No standalone lilbee build for $1/$2." >&2; return 1 ;;
  esac
}

# The sha256 of asset $2 in sums file $1, or nothing.
sum_of() { awk -v name="$2" '$2 == name { print $1; exit }' "$1"; }

# True when every gfx target in the comma list $1 is a line of manifest file $2.
gfx_shipped() {
  [ -s "$2" ] || return 1
  for gs_target in $(echo "$1" | tr ',' ' '); do
    grep -qx "${gs_target}" "$2" || return 1
  done
}

# The asset to install: os arch avx2 cuda kfd gfx rocm_userland forced_variant.
# SUMS_FILE lists the release assets. GFX_FILE lists the gfx targets the ROCm build ships.
select_asset() {
  sel_variant="$8"
  sel_gfx="-"
  if [ -n "${sel_variant}" ]; then
    asset_name "$1" "$2" "${sel_variant}" >/dev/null || return 1
  else
    sel_variant=$(host_variant "$1" "$2" "$3" "$4" "$5" "$6" "$7")
    sel_gfx="$6"
  fi
  [ "${sel_variant}" != default ] || sel_variant=""
  sel_baseline=default
  case "${sel_variant}" in compat | compat-*) sel_baseline=compat ;; esac
  sel_base_asset=$(asset_name "$1" "$2" "${sel_baseline}") || return 1
  if [ -z "$(sum_of "${SUMS_FILE}" "${sel_base_asset}")" ]; then
    echo "This release has no build for $1/$2." >&2
    return 1
  fi
  if [ -n "${sel_variant}" ] && [ "${sel_variant}" != "${sel_baseline}" ]; then
    sel_gpu_asset=$(asset_name "$1" "$2" "${sel_variant}") || return 1
    if [ -n "$(sum_of "${SUMS_FILE}" "${sel_gpu_asset}")" ]; then
      case "${sel_variant}" in
        *rocm) [ "${sel_gfx}" = "-" ] || gfx_shipped "${sel_gfx}" "${GFX_FILE}" || sel_gpu_asset="" ;;
      esac
      if [ -n "${sel_gpu_asset}" ]; then echo "${sel_gpu_asset}"; return 0; fi
    fi
  fi
  echo "${sel_base_asset}"
}

detect_os() {
  case "$(uname -s)" in
    Darwin) echo darwin ;;
    Linux) echo linux ;;
    *) die "this installer runs on macOS and Linux. On Windows, run: irm https://lilbee.sh/install.ps1 | iex" ;;
  esac
}

detect_arch() {
  case "$(uname -m)" in
    x86_64 | amd64)
      if [ "$1" = darwin ] && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || true)" = 1 ]; then echo arm64; else echo x64; fi
      ;;
    arm64 | aarch64) echo arm64 ;;
    *) uname -m ;;
  esac
}

# yes or no for AVX2 on x86-64. A CPU that cannot be read counts as yes.
detect_avx2() {
  [ "$2" = x64 ] || { echo yes; return 0; }
  case "$1" in
    darwin) if [ "$(sysctl -n hw.optional.avx2_0 2>/dev/null || echo 1)" = 0 ]; then echo no; else echo yes; fi ;;
    *)
      da_flags=$(grep -m1 '^flags' "${FS_ROOT}/proc/cpuinfo" 2>/dev/null || true)
      case "${da_flags}" in
        "") echo yes ;;
        *" avx2"*) echo yes ;;
        *) echo no ;;
      esac
      ;;
  esac
}

# The CUDA version the NVIDIA driver reports, such as 12.4, or "-".
detect_cuda() {
  dc_version=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1 || true)
  echo "${dc_version:--}"
}

# The gfx targets of the AMD GPUs the kernel reports, as a sorted comma list, or "-".
detect_gfx() {
  dg_names=""
  for dg_props in "${FS_ROOT}"/sys/class/kfd/kfd/topology/nodes/*/properties; do
    [ -r "${dg_props}" ] || continue
    dg_version=$(sed -n 's/^gfx_target_version[[:space:]]*\([0-9][0-9]*\)$/\1/p' "${dg_props}" 2>/dev/null || true)
    [ -n "${dg_version}" ] && [ "${dg_version}" -gt 0 ] || continue
    dg_names="${dg_names}$(printf 'gfx%d%x%x' $(( dg_version / 10000 )) $(( dg_version / 100 % 100 )) $(( dg_version % 100 ))) "
  done
  if [ -z "${dg_names}" ]; then echo "-"; else echo "${dg_names}" | tr ' ' '\n' | grep . | sort -u | paste -sd, -; fi
}

check_platform_floor() {
  case "$1" in
    darwin)
      cf_floor="${MACOS_FLOOR_X86_64}"
      [ "$2" != arm64 ] || cf_floor="${MACOS_FLOOR_ARM64}"
      cf_macos=$(sw_vers -productVersion 2>/dev/null || echo "${cf_floor}")
      version_ge "${cf_macos}" "${cf_floor}" || die "this build needs macOS ${cf_floor} or newer. This Mac runs ${cf_macos}."
      ;;
    linux)
      if ldd --version 2>&1 | grep -qi musl; then
        die "this Linux uses musl, and the lilbee binary needs glibc. ${OTHER_ROUTES}"
      fi
      cf_glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{ print $2 }')
      [ -z "${cf_glibc}" ] || version_ge "${cf_glibc}" "${GLIBC_FLOOR}" \
        || die "this build needs glibc ${GLIBC_FLOOR} or newer. This system has ${cf_glibc}. ${OTHER_ROUTES}"
      ;;
  esac
}

fetch() {
  if have curl; then curl -fsSL --retry 3 -o "$2" "$1"
  elif have wget; then wget -q --tries=3 -O "$2" "$1"
  else die "curl or wget is required to download lilbee."
  fi
}

sha256_of() {
  if have sha256sum; then sha256sum "$1" | awk '{ print $1 }'
  elif have shasum; then shasum -a 256 "$1" | awk '{ print $1 }'
  else die "sha256sum or shasum is required to verify the download."
  fi
}

# Put folder $1 on PATH for new shells, unless LILBEE_NO_MODIFY_PATH=1.
add_to_path() {
  if [ "${GITHUB_ACTIONS:-}" = true ] && [ -n "${GITHUB_PATH:-}" ]; then echo "$1" >> "${GITHUB_PATH}"; fi
  case ":${PATH}:" in *":$1:"*) return 0 ;; esac
  ap_line="export PATH=\"$1:\$PATH\""
  if [ "${LILBEE_NO_MODIFY_PATH:-}" = 1 ]; then
    say "add ${1} to your PATH: ${ap_line}"
    return 0
  fi
  case "${SHELL:-}" in
    */zsh) ap_rc="${ZDOTDIR:-${HOME}}/.zshrc" ;;
    */bash) if [ "$(uname -s)" = Darwin ]; then ap_rc="${HOME}/.bash_profile"; else ap_rc="${HOME}/.bashrc"; fi ;;
    */fish)
      ap_rc="${XDG_CONFIG_HOME:-${HOME}/.config}/fish/conf.d/lilbee.fish"
      ap_line="fish_add_path --global \"$1\""
      mkdir -p "$(dirname "${ap_rc}")"
      ;;
    *) ap_rc="${HOME}/.profile" ;;
  esac
  if ! grep -qxF "${ap_line}" "${ap_rc}" 2>/dev/null; then
    printf '\n# lilbee\n%s\n' "${ap_line}" >> "${ap_rc}"
  fi
  say "added ${1} to PATH in ${ap_rc}. Open a new terminal, or run: ${ap_line}"
}

# Download the ROCm gfx manifest of the wanted build into GFX_FILE when SHA256SUMS vouches for it.
fetch_gfx_manifest() {
  fg_name=$(asset_name "$1" "$2" "$3" 2>/dev/null) || return 0
  fg_name="${fg_name}.gfx.txt"
  fg_sum=$(sum_of "${SUMS_FILE}" "${fg_name}")
  [ -n "${fg_sum}" ] || return 0
  fetch "${base_url}/${fg_name}" "${GFX_FILE}" || return 0
  [ "$(sha256_of "${GFX_FILE}")" = "${fg_sum}" ] || : > "${GFX_FILE}"
}

# Hardware facts as globals: avx2, cuda, kfd, gfx, userland.
probe_hardware() {
  avx2=yes cuda="-" kfd=no gfx="-" userland=no
  [ -z "${LILBEE_VARIANT:-}" ] || return 0
  avx2=$(detect_avx2 "$1" "$2")
  [ "$1" = linux ] || return 0
  cuda=$(detect_cuda)
  [ -e "${FS_ROOT}/dev/kfd" ] || return 0
  kfd=yes
  gfx=$(detect_gfx)
  if [ "${gfx}" = "-" ] && { [ -e "${FS_ROOT}/opt/rocm" ] || rocminfo >/dev/null 2>&1; }; then userland=yes; fi
}

# Write the verified download to ${dest}.
install_file() {
  free_kb=$(df -Pk "${install_dir}" | awk 'NR == 2 { print $4 }')
  [ "${free_kb:-0}" -ge "${MIN_FREE_KB}" ] || die "not enough free space in ${install_dir}: lilbee needs about 2 GB."
  say "downloading ${asset}"
  partial=$(mktemp "${install_dir}/.lilbee.XXXXXX")
  fetch "${base_url}/${asset}" "${partial}" || die "the download of ${asset} failed."
  if [ "$(sha256_of "${partial}")" != "${expected}" ]; then
    die "the SHA-256 of ${asset} does not match SHA256SUMS. Nothing was installed."
  fi
  chmod 755 "${partial}"
  mv -f "${partial}" "${dest}"
  say "installed ${dest} (${asset}). The SHA-256 matches SHA256SUMS."
}

main() {
  os=$(detect_os)
  arch=$(detect_arch "${os}")
  case "${os}/${arch}" in
    darwin/x64 | darwin/arm64 | linux/x64) ;;
    linux/arm64) die "no standalone lilbee build for Linux arm64. ${OTHER_ROUTES}" ;;
    *) die "no lilbee build for ${os}/${arch}. ${NPM_HINT}" ;;
  esac
  probe_hardware "${os}" "${arch}"
  work=$(mktemp -d)
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  base_url="${RELEASES_URL}/latest/download"
  [ -z "${LILBEE_RELEASE:-}" ] || base_url="${RELEASES_URL}/download/${LILBEE_RELEASE}"
  SUMS_FILE="${work}/SHA256SUMS"
  GFX_FILE="${work}/gfx.txt"
  : > "${GFX_FILE}"
  fetch "${base_url}/SHA256SUMS" "${SUMS_FILE}" \
    || die "this release has no SHA256SUMS file, so lilbee cannot verify a download. ${NPM_HINT}"
  forced="${LILBEE_VARIANT:-}"
  wanted="${forced:-$(host_variant "${os}" "${arch}" "${avx2}" "${cuda}" "${kfd}" "${gfx}" "${userland}")}"
  case "${wanted}" in *rocm) [ "${gfx}" = "-" ] || fetch_gfx_manifest "${os}" "${arch}" "${wanted}" ;; esac
  asset=$(select_asset "${os}" "${arch}" "${avx2}" "${cuda}" "${kfd}" "${gfx}" "${userland}" "${forced}") \
    || die "no lilbee build for this machine in this release. ${NPM_HINT}"
  expected=$(sum_of "${SUMS_FILE}" "${asset}")
  check_platform_floor "${os}" "${arch}"
  install_dir="${LILBEE_INSTALL_DIR:-${XDG_BIN_HOME:-${HOME}/.local/bin}}"
  dest="${install_dir}/lilbee"
  mkdir -p "${install_dir}"
  if [ -f "${dest}" ] && [ "$(sha256_of "${dest}")" = "${expected}" ]; then
    say "lilbee is up to date at ${dest} (${asset})."
  else
    install_file
  fi
  add_to_path "${install_dir}"
  other=$(command -v lilbee 2>/dev/null || true)
  [ -z "${other}" ] || [ "${other}" = "${dest}" ] || warn "another lilbee comes first on PATH: ${other}"
  say "to uninstall, delete ${dest}. Models and data stay in place."
}

[ -n "${LILBEE_INSTALL_LIB:-}" ] || main "$@"
