#!/usr/bin/env bats
# scripts/publish_npm.sh against a copy of the release files, a fake npm and a
# fake registry script. The real release_version.sh runs the tag check.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  BIN="${FIXTURE}/bin"
  PKG="${FIXTURE}/pkg"
  CALLS="${FIXTURE}/calls"
  mkdir -p "${BIN}" "${PKG}/scripts" "${PKG}/packages/lilbee"
  : > "${CALLS}"
  cp "${REPO_ROOT}/scripts/publish_npm.sh" "${REPO_ROOT}/scripts/release_version.sh" "${PKG}/scripts/"

  # The registry script has its own bats file; here it only records the call.
  printf '#!/bin/sh\necho "registry $*" >> "%s"\n' "${CALLS}" > "${PKG}/scripts/publish_mcp_registry.sh"

  # npm view lists NPM_VERSIONS; npm publish records its arguments.
  cat > "${BIN}/npm" <<'STUB'
#!/bin/sh
case "$1" in
  view) echo "npm view $*" >> "${FAKE_CALLS}"; printf '%s\n' "${NPM_VERSIONS:-[]}" ;;
  publish) echo "npm $*" >> "${FAKE_CALLS}" ;;
  *) echo "npm $*" >> "${FAKE_CALLS}"; exit 1 ;;
esac
STUB
  chmod +x "${BIN}/npm"
}

# The release files for one tag: the package version, its lilbee.release and the listing version.
release_files() {  # package version, lilbee.release
  printf '{"version": "%s", "mcpName": "io.example/lilbee", "lilbee": {"release": "%s"}}\n' "$1" "$2" \
    > "${PKG}/packages/lilbee/package.json"
  printf '{"name": "io.example/lilbee", "version": "%s", "packages": [{"version": "%s"}]}\n' "$1" "$1" \
    > "${PKG}/server.json"
}

publish() {  # tag
  cd "${PKG}" && FAKE_CALLS="${CALLS}" PATH="${BIN}:${PATH}" bash scripts/publish_npm.sh "$1"
}

@test "a tag whose files match is published with the latest tag, then listed in the registry" {
  release_files 1.2.3-beta.1000 v1.2.3b1000
  NPM_TRUSTED_PUBLISHING=enabled run publish v1.2.3b1000
  [ "$status" -eq 0 ]
  [ "$(grep -c '^npm publish' "${CALLS}")" = 1 ]
  grep -qx 'npm publish --access public --tag latest' "${CALLS}"
  grep -qx 'registry server.json' "${CALLS}"
}

@test "the listing runs after the publish" {
  release_files 1.2.3-beta.1000 v1.2.3b1000
  NPM_TRUSTED_PUBLISHING=enabled run publish v1.2.3b1000
  [ "$status" -eq 0 ]
  [ "$(grep -n '^npm publish' "${CALLS}" | cut -d: -f1)" -lt "$(grep -n '^registry' "${CALLS}" | cut -d: -f1)" ]
}

@test "a package.json that is not the tag's release stops the job before anything is published or listed" {
  release_files 1.2.3-beta.691 v1.2.3b691
  NPM_TRUSTED_PUBLISHING=enabled run publish v1.2.3b1000
  [ "$status" -eq 1 ]
  [[ "$output" == *"needs version 1.2.3-beta.1000"* ]]
  ! grep -q '^npm publish' "${CALLS}"
  ! grep -q '^registry' "${CALLS}"
}

@test "a server.json that is not the tag's version stops the job before anything is published or listed" {
  release_files 1.2.3-beta.1000 v1.2.3b1000
  printf '{"name": "io.example/lilbee", "version": "1.2.3-beta.691", "packages": [{"version": "1.2.3-beta.691"}]}\n' \
    > "${PKG}/server.json"
  NPM_TRUSTED_PUBLISHING=enabled run publish v1.2.3b1000
  [ "$status" -eq 1 ]
  [[ "$output" == *"server.json does not carry version 1.2.3-beta.1000"* ]]
  ! grep -q '^npm publish' "${CALLS}"
  ! grep -q '^registry' "${CALLS}"
}

@test "a dev build is neither published nor listed, and the request fails" {
  release_files 1.2.3-beta.420.dev.711 v1.2.3b420.dev711
  NPM_TRUSTED_PUBLISHING=enabled run publish v1.2.3b420.dev711
  [ "$status" -eq 1 ]
  [[ "$output" == *"is a dev build"* ]]
  [ ! -s "${CALLS}" ]
}

# The resolve step that decides whether the npm job runs, as the workflow runs it.
npm_requested() {  # channels tag
  local wf="${REPO_ROOT}/.github/workflows/publish-packages.yml" out="${BATS_TEST_TMPDIR}/github_output"
  : > "${out}"
  step=$(yq -r '.jobs.resolve.steps[] | select(.id == "npm") | .run' "${wf}")
  [ -n "${step}" ] || return 2
  # The step's log line goes to the job log; only its output counts here.
  CHANNELS="$1" TAG="$2" GITHUB_OUTPUT="${out}" bash -c "${step}" > /dev/null || return
  sed -n 's/^requested=//p' "${out}"
}

@test "channels all does not request npm for a dev tag, and naming npm still does" {
  run npm_requested all v1.2.3b420.dev711
  [ "$output" = false ]
  run npm_requested all v1.2.3b1000
  [ "$output" = true ]
  run npm_requested npm v1.2.3b420.dev711
  [ "$output" = true ]
  run npm_requested homebrew,npm v1.2.3b1000
  [ "$output" = true ]
  run npm_requested homebrew,nix v1.2.3b1000
  [ "$output" = false ]
  # The npm job runs on that decision and on nothing else.
  run yq -r '.jobs.npm.if' "${REPO_ROOT}/.github/workflows/publish-packages.yml"
  [ "$output" = "needs.resolve.outputs.npm == 'true'" ]
  run yq -r '.jobs.resolve.outputs.npm' "${REPO_ROOT}/.github/workflows/publish-packages.yml"
  [ "$output" = '${{ steps.npm.outputs.requested }}' ]
}

@test "a version npm already holds is not published again, and the listing still runs" {
  release_files 1.2.3-beta.1000 v1.2.3b1000
  NPM_VERSIONS='["1.2.2","1.2.3-beta.1000"]' NPM_TRUSTED_PUBLISHING=enabled run publish v1.2.3b1000
  [ "$status" -eq 0 ]
  [[ "$output" == *"lilbee 1.2.3-beta.1000 on npm: true"* ]]
  [[ "$output" == *"already on npm"* ]]
  ! grep -q '^npm publish' "${CALLS}"
  grep -qx 'registry server.json' "${CALLS}"
}

@test "npm answering with one version as a bare string is read as a list of one" {
  release_files 1.2.3-beta.1000 v1.2.3b1000
  NPM_VERSIONS='"1.2.3-beta.1000"' NPM_TRUSTED_PUBLISHING=enabled run publish v1.2.3b1000
  [ "$status" -eq 0 ]
  ! grep -q '^npm publish' "${CALLS}"
}

@test "a version npm lacks is not read as published because another beta is" {
  release_files 1.2.3-beta.1000 v1.2.3b1000
  NPM_VERSIONS='["1.2.3-beta.100"]' NPM_TRUSTED_PUBLISHING=enabled run publish v1.2.3b1000
  [ "$status" -eq 0 ]
  grep -qx 'npm publish --access public --tag latest' "${CALLS}"
}

@test "without trusted publishing the job publishes nothing, says why, and fails" {
  release_files 1.2.3-beta.1000 v1.2.3b1000
  NPM_TRUSTED_PUBLISHING= run publish v1.2.3b1000
  [ "$status" -eq 1 ]
  [[ "$output" == *"NPM_TRUSTED_PUBLISHING is not 'enabled'"* ]]
  ! grep -q '^npm publish' "${CALLS}"
  ! grep -q '^registry' "${CALLS}"
}

# The npm job's publish step as the workflow runs it, with the repository
# variable missing: a vars value that does not exist expands to empty.
@test "the workflow's publish step fails when the repository variable is missing" {
  release_files 1.2.3-beta.1000 v1.2.3b1000
  step=$(yq -r '.jobs.npm.steps[] | select(.name == "Publish lilbee and list it in the MCP Registry") | .run' \
    "${REPO_ROOT}/.github/workflows/publish-packages.yml")
  [ -n "${step}" ]
  cd "${PKG}"
  FAKE_CALLS="${CALLS}" PATH="${BIN}:${PATH}" TAG=v1.2.3b1000 NPM_TRUSTED_PUBLISHING= run bash -c "${step}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"did not publish"* ]]
  ! grep -q '^npm publish' "${CALLS}"
}

@test "without trusted publishing a package that is not the tag's release still stops the job" {
  release_files 1.2.3-beta.691 v1.2.3b691
  NPM_TRUSTED_PUBLISHING= run publish v1.2.3b1000
  [ "$status" -eq 1 ]
}

@test "a failed registry listing fails the job" {
  release_files 1.2.3-beta.1000 v1.2.3b1000
  printf '#!/bin/sh\nexit 7\n' > "${PKG}/scripts/publish_mcp_registry.sh"
  NPM_TRUSTED_PUBLISHING=enabled run publish v1.2.3b1000
  [ "$status" -eq 7 ]
}
