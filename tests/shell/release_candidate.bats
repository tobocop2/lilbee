#!/usr/bin/env bats
# The release-candidate resolve shell against stubbed git and gh, plus the job
# graph the dispatch cascade depends on.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  CANDIDATE="${REPO_ROOT}/.github/workflows/release-candidate.yml"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  mkdir -p "${FIXTURE}/bin" "${FIXTURE}/releases" "${FIXTURE}/release_body"
  : > "${FIXTURE}/actions.log"
  printf 'parent-sha\n' > "${FIXTURE}/parent"
  : > "${FIXTURE}/ls_remote"
  echo 0 > "${FIXTURE}/diff_rc"
  printf '[project]\nversion = "1.2.3"\n' > "${FIXTURE}/pyproject.toml"
  cp "${BATS_TEST_DIRNAME}/stubs/git-candidate" "${FIXTURE}/bin/git"
  cp "${BATS_TEST_DIRNAME}/stubs/gh-candidate" "${FIXTURE}/bin/gh"
  chmod +x "${FIXTURE}/bin/git" "${FIXTURE}/bin/gh"
}

compute() {  # event ref-name github-ref [input-ref]
  bash "${REPO_ROOT}/scripts/compute_version_tag.sh" \
    "${FIXTURE}/pyproject.toml" "$@" >> "${FIXTURE}/outputs"
}

detect() {  # tag
  PATH="${FIXTURE}/bin:${PATH}" FIXTURE="${FIXTURE}" \
    bash "${REPO_ROOT}/scripts/detect_promotion.sh" "$@" >> "${FIXTURE}/outputs"
}

prerelease() {  # tag version repo [from]
  PATH="${FIXTURE}/bin:${PATH}" FIXTURE="${FIXTURE}" \
    bash "${REPO_ROOT}/scripts/create_prerelease.sh" "$@"
}

@test "a push tag matching the version resolves tag and ref" {
  run compute push v1.2.3 refs/tags/v1.2.3
  [ "$status" -eq 0 ]
  [ "$(cat "${FIXTURE}/outputs")" = "$(printf 'version=1.2.3\ntag=v1.2.3\nref=refs/tags/v1.2.3')" ]
}

@test "a push tag without the v prefix resolves" {
  run compute push 1.2.3 refs/tags/1.2.3
  [ "$status" -eq 0 ]
  grep -q '^tag=1.2.3$' "${FIXTURE}/outputs"
}

@test "a push tag that mismatches the version fails" {
  run compute push v9.9.9 refs/tags/v9.9.9
  [ "$status" -eq 1 ]
  [[ "$output" == *"does not match"* ]]
}

@test "a dispatch run defaults the ref to the branch" {
  run compute workflow_dispatch ignored refs/heads/main
  [ "$status" -eq 0 ]
  [ "$(cat "${FIXTURE}/outputs")" = "$(printf 'version=1.2.3\ntag=v1.2.3\nref=refs/heads/main')" ]
}

@test "a dispatch run honors an explicit input ref" {
  run compute workflow_dispatch ignored refs/heads/main refs/heads/hotfix
  [ "$status" -eq 0 ]
  grep -q '^ref=refs/heads/hotfix$' "${FIXTURE}/outputs"
}

@test "a pyproject without a version fails" {
  printf '[project]\n' > "${FIXTURE}/pyproject.toml"
  run compute push v1.2.3 refs/tags/v1.2.3
  [ "$status" -eq 1 ]
  [[ "$output" == *"could not parse version"* ]]
}

@test "a version-only change on a tagged parent is a promotion" {
  printf 'abc123 refs/tags/v1.2.2\n' > "${FIXTURE}/ls_remote"
  printf 'abc123\n' > "${FIXTURE}/parent"
  run detect v1.2.3
  [ "$status" -eq 0 ]
  [ "$(cat "${FIXTURE}/outputs")" = "promoted_from=v1.2.2" ]
  [[ "$output" == *"same-source promotion"* ]]
}

@test "an annotated-tag peel does not promote" {
  printf 'abc123 refs/tags/v1.2.2^{}\n' > "${FIXTURE}/ls_remote"
  printf 'abc123\n' > "${FIXTURE}/parent"
  run detect v1.2.3
  [ "$status" -eq 0 ]
  [ "$(cat "${FIXTURE}/outputs")" = "promoted_from=" ]
}

@test "a change outside the version files is not a promotion" {
  printf 'abc123 refs/tags/v1.2.2\n' > "${FIXTURE}/ls_remote"
  printf 'abc123\n' > "${FIXTURE}/parent"
  echo 1 > "${FIXTURE}/diff_rc"
  run detect v1.2.3
  [ "$status" -eq 0 ]
  [ "$(cat "${FIXTURE}/outputs")" = "promoted_from=" ]
}

@test "a parent that is not a tag is not a promotion" {
  printf 'abc123 refs/heads/main\n' > "${FIXTURE}/ls_remote"
  printf 'abc123\n' > "${FIXTURE}/parent"
  run detect v1.2.3
  [ "$status" -eq 0 ]
  [ "$(cat "${FIXTURE}/outputs")" = "promoted_from=" ]
}

@test "a tag equal to its parent tag is not a promotion" {
  printf 'abc123 refs/tags/v1.2.3\n' > "${FIXTURE}/ls_remote"
  printf 'abc123\n' > "${FIXTURE}/parent"
  run detect v1.2.3
  [ "$status" -eq 0 ]
  [ "$(cat "${FIXTURE}/outputs")" = "promoted_from=" ]
}

@test "a root commit with no parent is not a promotion" {
  rm "${FIXTURE}/parent"
  run detect v1.2.3
  [ "$status" -eq 0 ]
  [ "$(cat "${FIXTURE}/outputs")" = "promoted_from=" ]
}

@test "an existing pre-release is left alone" {
  : > "${FIXTURE}/releases/v1.2.3"
  run prerelease v1.2.3 1.2.3 tobocop2/lilbee
  [ "$status" -eq 0 ]
  [ ! -f "${FIXTURE}/created_notes" ]
  [ "$(cat "${FIXTURE}/actions.log")" = "view v1.2.3" ]
}

@test "a promotion carries the source notes under a banner" {
  printf 'source notes\n' > "${FIXTURE}/release_body/v1.2.2"
  run prerelease v1.2.3 1.2.3 tobocop2/lilbee v1.2.2
  [ "$status" -eq 0 ]
  [ "$(cat "${FIXTURE}/created_notes")" = "$(printf 'Promoted from [v1.2.2](https://github.com/tobocop2/lilbee/releases/tag/v1.2.2): the same source, released as 1.2.3.\n\nsource notes')" ]
}

@test "a promotion whose source notes are gone falls back to the default notes" {
  run prerelease v1.2.3 1.2.3 tobocop2/lilbee v1.2.2
  [ "$status" -eq 0 ]
  grep -q 'Release candidate building' "${FIXTURE}/created_notes"
}

@test "a normal release gets the default notes" {
  run prerelease v1.2.3 1.2.3 tobocop2/lilbee
  [ "$status" -eq 0 ]
  grep -q 'Release candidate building' "${FIXTURE}/created_notes"
}

needs_of() {  # job
  yq -r "(.jobs.\"$1\".needs // [])[]" "${CANDIDATE}"
}

@test "every dispatch job is gated on the build workflow it ships assets from" {
  # A dispatch that outruns its assets starts a publisher that skips the
  # missing one and still concludes success, which no rerun corrects. Only the
  # implicit success() on a `needs` entry makes the dispatch skip instead.
  # attach-prerelease is excluded from the walk: it is the one job that runs on
  # purpose with a cell missing, so reaching a build workflow through it gates
  # nothing.
  local jobs job direct reach hop
  jobs=$(yq -r '.jobs | keys | .[] | select(test("^dispatch-"))' "${CANDIDATE}")
  [ "$(echo "${jobs}" | wc -l)" -eq 5 ]
  while IFS= read -r job; do
    direct=$(echo "$(needs_of "${job}")" | grep -v '^attach-prerelease$' || true)
    reach="${direct}"
    while IFS= read -r hop; do
      [ -n "${hop}" ] && reach="${reach}"$'\n'"$(needs_of "${hop}")"
    done <<< "${direct}"
    echo "${job} is gated by: ${reach}" >&2
    echo "${reach}" | grep -q '^build-'
  done <<< "${jobs}"
}
