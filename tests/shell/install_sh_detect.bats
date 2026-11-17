#!/usr/bin/env bats
# site/install.sh: what it reads from a machine (CPU flags, driver banner, KFD topology, ROCm folder)
# for every machine case in the case table. Each case builds a fake root and stub tools.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  PROBE="${REPO_ROOT}/tests/installer/probe_hardware.sh"
  CASES="${REPO_ROOT}/tests/installer/cases.tsv"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  mkdir -p "${FIXTURE}/bin"
  for stub in nvidia-smi sysctl rocminfo; do cp "${BATS_TEST_DIRNAME}/stubs/${stub}" "${FIXTURE}/bin/${stub}"; done
  PATH="${FIXTURE}/bin:${PATH}"
  unset LILBEE_VARIANT NVIDIA_CUDA SYSCTL_AVX2 SYSCTL_TRANSLATED ROCMINFO_OK
  machines=0
}

# gfx1100 -> 110000, gfx906 -> 90006, gfx90a -> 90010: the KFD gfx_target_version of a gfx name.
gfx_version() {
  g="${1#gfx}"
  step="${g: -1}"
  rest="${g%?}"
  minor="${rest: -1}"
  major="${rest%?}"
  echo $((major * 10000 + 0x${minor} * 100 + 0x${step}))
}

# A fresh fake root in $root with a CPU node and one KFD node per gfx name in $1 (comma list or -).
fake_root() {
  machines=$((machines + 1))
  root="${BATS_TEST_TMPDIR}/root${machines}"
  mkdir -p "${root}/proc" "${root}/dev" "${root}/opt" "${root}/sys/class/kfd/kfd/topology/nodes/0"
  printf 'cpu_cores_count 8\ngfx_target_version 0\n' > "${root}/sys/class/kfd/kfd/topology/nodes/0/properties"
  n=0
  if [ "$1" != "-" ]; then
    for target in $(echo "$1" | tr ',' ' '); do
      n=$((n + 1))
      mkdir -p "${root}/sys/class/kfd/kfd/topology/nodes/${n}"
      printf 'cpu_cores_count 0\ngfx_target_version %s\n' "$(gfx_version "${target}")" > "${root}/sys/class/kfd/kfd/topology/nodes/${n}/properties"
    done
  fi
}

cpuinfo() {
  if [ "$1" = yes ]; then
    printf 'processor\t: 0\nflags\t\t: fpu vme sse4_2 avx avx2 bmi2 fma\n' > "${root}/proc/cpuinfo"
  else
    printf 'processor\t: 0\nflags\t\t: fpu vme sse4_2 avx fma\n' > "${root}/proc/cpuinfo"
  fi
}

probe() { run env LILBEE_FS_ROOT="${root}" sh "${PROBE}" "$@"; }

@test "every machine case: the facts install.sh reads from the machine are the facts of its case row" {
  total=0
  matched=0
  while read -r id os arch avx2 cuda kfd gfx userland; do
    [ "${os}" != win32 ] || continue
    total=$((total + 1))
    fake_root "${gfx}"
    cpuinfo "${avx2}"
    [ "${kfd}" != yes ] || : > "${root}/dev/kfd"
    [ "${userland}" != yes ] || mkdir -p "${root}/opt/rocm"
    want="avx2=${avx2} cuda=${cuda} kfd=${kfd} gfx=${gfx} userland=${userland}"
    # The stub nvidia-smi fails, as the real one does without a driver, unless NVIDIA_CUDA is set.
    [ "${cuda}" = "-" ] && unset NVIDIA_CUDA || export NVIDIA_CUDA="${cuda}"
    [ "${avx2}" = no ] && export SYSCTL_AVX2=0 || export SYSCTL_AVX2=1
    got=$(LILBEE_FS_ROOT="${root}" sh "${PROBE}" "${os}" "${arch}" 2>&1) || got="FAILED ${got}"
    if [ "${got}" = "${want}" ]; then matched=$((matched + 1)); else echo "${id}: expected ${want}, got ${got}" >&3; fi
  done < <(awk -F'\t' 'NR > 1 && $9 == "" { print $1, $2, $3, $4, $5, $6, $7, $8 }' "${CASES}")
  echo "matched ${matched} of ${total}" >&3
  [ "${total}" -gt 100 ]
  [ "${matched}" -eq "${total}" ]
}

@test "a CPU without avx2 in its flags is read as no, and one with it as yes" {
  fake_root -; cpuinfo no
  probe linux x64
  [[ "$output" == "avx2=no "* ]]
  cpuinfo yes
  probe linux x64
  [[ "$output" == "avx2=yes "* ]]
}

@test "a CPU that cannot be read counts as avx2 yes" {
  fake_root -
  probe linux x64
  [[ "$output" == "avx2=yes "* ]]
}

@test "an arm64 host is not asked for avx2" {
  fake_root -; cpuinfo no
  probe linux arm64
  [[ "$output" == "avx2=yes "* ]]
}

@test "a Mac without avx2 reads the sysctl answer" {
  fake_root -
  SYSCTL_AVX2=0 probe darwin x64
  [[ "$output" == "avx2=no cuda=- kfd=no"* ]]
  SYSCTL_AVX2=1 probe darwin x64
  [[ "$output" == "avx2=yes cuda=- kfd=no"* ]]
}

@test "the CUDA version comes from the driver banner, whatever the other banner numbers are" {
  fake_root -
  NVIDIA_CUDA=12.4 probe linux x64
  [[ "$output" == *" cuda=12.4 "* ]]
  NVIDIA_CUDA=11.8 probe linux x64
  [[ "$output" == *" cuda=11.8 "* ]]
}

@test "a failing nvidia-smi leaves the CUDA version at none" {
  fake_root -
  probe linux x64
  [[ "$output" == *" cuda=- "* ]]
}

@test "a gfx name takes a hex minor and a hex step from the KFD version" {
  fake_root gfx90a
  : > "${root}/dev/kfd"
  probe linux x64
  [[ "$output" == *" kfd=yes gfx=gfx90a "* ]]
  fake_root -
  mkdir -p "${root}/sys/class/kfd/kfd/topology/nodes/1"
  printf 'gfx_target_version 101200\n' > "${root}/sys/class/kfd/kfd/topology/nodes/1/properties"
  : > "${root}/dev/kfd"
  probe linux x64
  [[ "$output" == *" gfx=gfx10c0 "* ]]
}

@test "two nodes with the same gfx name give one name, and names come out sorted" {
  fake_root gfx1100,gfx906,gfx1100
  : > "${root}/dev/kfd"
  probe linux x64
  [[ "$output" == *" gfx=gfx1100,gfx906 "* ]]
}

@test "an unreadable KFD node does not end the run and the other nodes are still read" {
  fake_root -
  mkdir -p "${root}/sys/class/kfd/kfd/topology/nodes/1/properties" "${root}/sys/class/kfd/kfd/topology/nodes/2"
  printf 'gfx_target_version 110000\n' > "${root}/sys/class/kfd/kfd/topology/nodes/2/properties"
  : > "${root}/dev/kfd"
  probe linux x64
  [ "$status" -eq 0 ]
  [[ "$output" == *" kfd=yes gfx=gfx1100 "* ]]
}

@test "without /dev/kfd no AMD fact is read, even when the topology lists a GPU" {
  fake_root gfx1100
  probe linux x64
  [[ "$output" == *" kfd=no gfx=- userland=no" ]]
}

@test "/dev/kfd with no GPU node and no ROCm folder is kfd yes and userland no" {
  fake_root -
  : > "${root}/dev/kfd"
  probe linux x64
  [[ "$output" == *" kfd=yes gfx=- userland=no" ]]
}

@test "rocminfo that works marks the ROCm userland present" {
  fake_root -
  : > "${root}/dev/kfd"
  ROCMINFO_OK=1 probe linux x64
  [[ "$output" == *" kfd=yes gfx=- userland=yes" ]]
}

@test "a gfx name found means the userland is not probed" {
  fake_root gfx1100
  : > "${root}/dev/kfd"
  mkdir -p "${root}/opt/rocm"
  probe linux x64
  [[ "$output" == *" gfx=gfx1100 userland=no" ]]
}

@test "LILBEE_VARIANT skips every hardware probe" {
  fake_root gfx1100; cpuinfo no
  : > "${root}/dev/kfd"
  run env LILBEE_VARIANT=cu124 NVIDIA_CUDA=12.4 LILBEE_FS_ROOT="${root}" sh "${PROBE}" linux x64
  [ "$output" = "avx2=yes cuda=- kfd=no gfx=- userland=no" ]
}

# The install flow with detection on: a ROCm host whose gfx manifest the release publishes.

flow_setup() {
  mkdir -p "${FIXTURE}/release" "${BATS_TEST_TMPDIR}/home"
  : > "${FIXTURE}/curl.log"
  cp "${BATS_TEST_DIRNAME}/stubs/curl-release" "${FIXTURE}/bin/curl"
  for stub in uname getconf ldd; do cp "${BATS_TEST_DIRNAME}/stubs/${stub}" "${FIXTURE}/bin/${stub}"; done
  export FIXTURE HOME="${BATS_TEST_TMPDIR}/home" SHELL=/bin/zsh LILBEE_INSTALL_DIR="${BATS_TEST_TMPDIR}/home/bin"
  export LILBEE_NO_MODIFY_PATH=1 UNAME_S=Linux UNAME_M=x86_64
  unset LILBEE_RELEASE GITHUB_ACTIONS GITHUB_PATH XDG_BIN_HOME
}

hash_of() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# Publish asset $1 with body $2 and a SHA256SUMS line that matches it, or $3 as the hash.
publish() {
  printf '%s' "$2" > "${FIXTURE}/release/$1"
  printf '%s  %s\n' "${3:-$(hash_of "${FIXTURE}/release/$1")}" "$1" >> "${FIXTURE}/release/SHA256SUMS"
}

@test "a ROCm host whose gfx is in the manifest installs the ROCm build" {
  flow_setup
  fake_root gfx1100; cpuinfo yes
  : > "${root}/dev/kfd"
  publish lilbee-linux-x86_64 "plain"
  publish lilbee-linux-x86_64-rocm "rocm"
  publish lilbee-linux-x86_64-rocm.gfx.txt $'gfx1100\ngfx90a\n'
  run env LILBEE_FS_ROOT="${root}" sh "${REPO_ROOT}/site/install.sh"
  [ "$status" -eq 0 ]
  [ "$(cat "${LILBEE_INSTALL_DIR}/lilbee")" = "rocm" ]
}

@test "a gfx manifest whose hash is not in SHA256SUMS is ignored and the plain build installs" {
  flow_setup
  fake_root gfx1100; cpuinfo yes
  : > "${root}/dev/kfd"
  publish lilbee-linux-x86_64 "plain"
  publish lilbee-linux-x86_64-rocm "rocm"
  publish lilbee-linux-x86_64-rocm.gfx.txt $'gfx1100\n' "$(printf 'x' | shasum -a 256 | cut -d' ' -f1)"
  run env LILBEE_FS_ROOT="${root}" sh "${REPO_ROOT}/site/install.sh"
  [ "$status" -eq 0 ]
  [ "$(cat "${LILBEE_INSTALL_DIR}/lilbee")" = "plain" ]
}

@test "a ROCm host whose gfx is not in the manifest installs the plain build" {
  flow_setup
  fake_root gfx906; cpuinfo yes
  : > "${root}/dev/kfd"
  publish lilbee-linux-x86_64 "plain"
  publish lilbee-linux-x86_64-rocm "rocm"
  publish lilbee-linux-x86_64-rocm.gfx.txt $'gfx1100\n'
  run env LILBEE_FS_ROOT="${root}" sh "${REPO_ROOT}/site/install.sh"
  [ "$status" -eq 0 ]
  [ "$(cat "${LILBEE_INSTALL_DIR}/lilbee")" = "plain" ]
}

@test "a CPU without avx2 installs the compat build" {
  flow_setup
  fake_root -; cpuinfo no
  publish lilbee-linux-x86_64 "plain"
  publish lilbee-compat-linux-x86_64 "compat"
  run env LILBEE_FS_ROOT="${root}" sh "${REPO_ROOT}/site/install.sh"
  [ "$status" -eq 0 ]
  [ "$(cat "${LILBEE_INSTALL_DIR}/lilbee")" = "compat" ]
}

@test "a CUDA 12.5 driver installs the cu125 build" {
  flow_setup
  fake_root -; cpuinfo yes
  publish lilbee-linux-x86_64 "plain"
  publish lilbee-linux-x86_64-cu125 "cuda"
  run env NVIDIA_CUDA=12.5 LILBEE_FS_ROOT="${root}" sh "${REPO_ROOT}/site/install.sh"
  [ "$status" -eq 0 ]
  [ "$(cat "${LILBEE_INSTALL_DIR}/lilbee")" = "cuda" ]
}
