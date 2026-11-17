#!/usr/bin/env bats
# scripts/release_version.sh: the mapping from a lilbee version to the npm
# version, and the release files committed in this tree.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/release_version.sh"
  PACKAGE="${REPO_ROOT}/packages/lilbee"
}

pyproject_version() {
  sed -n 's/^version = "\(.*\)"$/\1/p' "${REPO_ROOT}/pyproject.toml" | head -1
}

@test "the committed package carries the lilbee version of pyproject in semver form" {
  version=$(pyproject_version)
  [ -n "${version}" ]
  [ "$(jq -r '.version' "${PACKAGE}/package.json")" = "$(bash "${SCRIPT}" npm-version "${version}")" ]
  [ "$(jq -r '.lilbee.release' "${PACKAGE}/package.json")" = "v${version}" ]
}

@test "the publish check accepts the committed release files for the pyproject version" {
  cd "${REPO_ROOT}"
  run bash "${SCRIPT}" check "v$(pyproject_version)"
  [ "$status" -eq 0 ]
}

@test "the lock file carries the package version in both version fields" {
  version=$(jq -r '.version' "${PACKAGE}/package.json")
  [ -n "${version}" ]
  [ "$(jq -r '.version' "${PACKAGE}/package-lock.json")" = "${version}" ]
  [ "$(jq -r '.packages[""].version' "${PACKAGE}/package-lock.json")" = "${version}" ]
}

@test "the beta number becomes a numeric prerelease identifier" {
  [ "$(bash "${SCRIPT}" npm-version 0.6.90b691)" = "0.6.90-beta.691" ]
  [ "$(bash "${SCRIPT}" npm-version 0.6.90b1000)" = "0.6.90-beta.1000" ]
  [ "$(bash "${SCRIPT}" npm-version 0.6.90)" = "0.6.90" ]
  [ "$(bash "${SCRIPT}" npm-version 0.6.90b420.dev711)" = "0.6.90-beta.420.dev.711" ]
}

@test "a four digit beta sorts after a three digit beta" {
  older=$(bash "${SCRIPT}" npm-version 0.6.90b691)
  newer=$(bash "${SCRIPT}" npm-version 0.6.90b1000)
  # The text before the number is equal, so version sort compares the numbers.
  [ "$(printf '%s\n%s\n' "${newer}" "${older}" | sort -V | head -1)" = "${older}" ]
  [ "${older%.*}" = "${newer%.*}" ]
}

@test "a version with no npm mapping is refused with its name" {
  for version in 0.7.0rc1 0.7.0a1 0.7.0b1.post1 0.7 v0.7.0; do
    run bash "${SCRIPT}" npm-version "${version}"
    [ "$status" -eq 1 ]
    [[ "$output" == *"'${version}' has no npm version"* ]]
  done
}

@test "every file a release stamps exists and the npm package is one of them" {
  cd "${REPO_ROOT}"
  listed=$(bash "${SCRIPT}" files)
  [ -n "${listed}" ]
  grep -qx 'packages/lilbee/package.json' <<< "${listed}"
  grep -qx 'packages/lilbee/package-lock.json' <<< "${listed}"
  while read -r name; do
    [ -f "${name}" ] || { echo "missing: ${name}"; return 1; }
  done <<< "${listed}"
}
