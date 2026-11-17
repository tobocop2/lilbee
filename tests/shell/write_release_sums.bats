#!/usr/bin/env bats
# scripts/write_release_sums.sh against a stubbed gh.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/write_release_sums.sh"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  export FIXTURE
  mkdir -p "${FIXTURE}/bin"
  : > "${FIXTURE}/actions.log"
  cp "${BATS_TEST_DIRNAME}/stubs/gh-sums" "${FIXTURE}/bin/gh"
  PATH="${FIXTURE}/bin:${PATH}"
  PRERELEASE=true
}

asset() {  # name digest-or-empty
  if [ -n "$2" ]; then printf '{"name":"%s","digest":"sha256:%s"}' "$1" "$2"; else printf '{"name":"%s","digest":null}' "$1"; fi
}

release() {  # asset JSON objects; PRERELEASE says whether the release is a prerelease
  local IFS=,
  printf '{"isPrerelease":%s,"assets":[%s]}\n' "${PRERELEASE}" "$*" > "${FIXTURE}/assets.json"
}

@test "each binary and gfx manifest gets a sha256sum line from its recorded digest" {
  release "$(asset lilbee-linux-x86_64 aaa)" "$(asset lilbee-linux-x86_64-rocm.gfx.txt bbb)" "$(asset lilbee-windows-x86_64.exe ccc)" "$(asset lilbee-macos-arm64 ddd)"
  run bash "${SCRIPT}" v1.0.0 owner/repo
  [ "$status" -eq 0 ]
  [ "$(cat "${FIXTURE}/uploaded/SHA256SUMS")" = "$(printf 'aaa  lilbee-linux-x86_64\nbbb  lilbee-linux-x86_64-rocm.gfx.txt\nddd  lilbee-macos-arm64\nccc  lilbee-windows-x86_64.exe')" ]
  grep -q 'upload v1.0.0 .*SHA256SUMS --repo owner/repo$' "${FIXTURE}/actions.log"
  ! grep -q -e '--clobber' "${FIXTURE}/actions.log"
}

@test "wheels, snaps and flatpakrefs are not listed" {
  release "$(asset lilbee-linux-x86_64 aaa)" "$(asset lilbee-linux-x86_64.snap eee)" "$(asset lilbee-0.6.1-py3-none-any.whl fff)" "$(asset lilbee.flatpakref ggg)" "$(asset openapi.json hhh)"
  run bash "${SCRIPT}" v1.0.0
  [ "$status" -eq 0 ]
  [ "$(cat "${FIXTURE}/uploaded/SHA256SUMS")" = "aaa  lilbee-linux-x86_64" ]
}

@test "a binary with no recorded digest fails and uploads nothing" {
  release "$(asset lilbee-linux-x86_64 aaa)" "$(asset lilbee-macos-arm64 '')"
  run bash "${SCRIPT}" v1.0.0
  [ "$status" -ne 0 ]
  [[ "$output" == *"no sha256 digest for: lilbee-macos-arm64"* ]]
  [ ! -e "${FIXTURE}/uploaded/SHA256SUMS" ]
}

@test "a release with no binaries fails and uploads nothing" {
  release "$(asset openapi.json hhh)"
  run bash "${SCRIPT}" v1.0.0
  [ "$status" -ne 0 ]
  [[ "$output" == *"no binary assets to list"* ]]
  [ ! -e "${FIXTURE}/uploaded/SHA256SUMS" ]
}

@test "a release that already has a matching SHA256SUMS is not written" {
  release "$(asset lilbee-linux-x86_64 aaa)" "$(asset lilbee-macos-arm64 ddd)" "$(asset SHA256SUMS '')"
  printf 'ddd  lilbee-macos-arm64\naaa  lilbee-linux-x86_64\n' > "${FIXTURE}/existing-sums"
  run bash "${SCRIPT}" v1.0.0
  [ "$status" -eq 0 ]
  [[ "$output" == *"matches"* ]]
  [ ! -s "${FIXTURE}/actions.log" ]
}

@test "a SHA256SUMS that disagrees with the digests fails and is not written" {
  release "$(asset lilbee-linux-x86_64 aaa)" "$(asset SHA256SUMS '')"
  printf 'bad  lilbee-linux-x86_64\n' > "${FIXTURE}/existing-sums"
  run bash "${SCRIPT}" v1.0.0
  [ "$status" -ne 0 ]
  [[ "$output" == *"disagrees with the recorded digests"* ]]
  [ ! -s "${FIXTURE}/actions.log" ]
}

@test "a published release without SHA256SUMS is left alone" {
  PRERELEASE=false
  release "$(asset lilbee-linux-x86_64 aaa)"
  run bash "${SCRIPT}" v1.0.0
  [ "$status" -eq 0 ]
  [[ "$output" == *"is published; SHA256SUMS left alone"* ]]
  [ ! -s "${FIXTURE}/actions.log" ]
}

@test "a published release with a matching SHA256SUMS is not written" {
  PRERELEASE=false
  release "$(asset lilbee-linux-x86_64 aaa)" "$(asset SHA256SUMS '')"
  printf 'aaa  lilbee-linux-x86_64\n' > "${FIXTURE}/existing-sums"
  run bash "${SCRIPT}" v1.0.0
  [ "$status" -eq 0 ]
  [ ! -s "${FIXTURE}/actions.log" ]
}

@test "a missing tag is a usage error" {
  run bash "${SCRIPT}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"usage:"* ]]
}
