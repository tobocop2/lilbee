#!/usr/bin/env bats
# scripts/release.sh and scripts/promote_release.sh against a throwaway repo and
# a bare origin. Real git, not the stub: the assertion is the tag object type.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  ORIGIN="${FIXTURE}/origin.git"
  WORK="${FIXTURE}/work"
  mkdir -p "${FIXTURE}/bin"
  # Only the throwaway repo's own config may decide the tag's shape.
  export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null
  export EDITOR_LOG="${FIXTURE}/editor.log"
  : > "${EDITOR_LOG}"
  cat > "${FIXTURE}/bin/editor" <<'EDITOR'
#!/usr/bin/env bash
echo "opened ${1:-}" >> "${EDITOR_LOG}"
exit 1
EDITOR
  chmod +x "${FIXTURE}/bin/editor"
  export GIT_EDITOR="${FIXTURE}/bin/editor"
}

make_repo() {  # tag.gpgsign value
  git init -q --bare "${ORIGIN}"
  git init -q -b main "${WORK}"
  git -C "${WORK}" config user.name "Release Test"
  git -C "${WORK}" config user.email "release@example.com"
  git -C "${WORK}" config commit.gpgsign false
  git -C "${WORK}" config tag.gpgsign "$1"
  printf 'version = "1.2.3b7"\n' > "${WORK}/pyproject.toml"
  printf 'version = "1.2.3b7"\n' > "${WORK}/uv.lock"
  git -C "${WORK}" add pyproject.toml uv.lock
  git -C "${WORK}" commit -q -m "Initial"
  git -C "${WORK}" remote add origin "${ORIGIN}"
  git -C "${WORK}" push -q -u origin main
}

seed_source_tag() {  # the tag promote_release.sh re-releases from
  git -C "${WORK}" -c tag.gpgsign=false tag v1.2.3b7
  git -C "${WORK}" push -q origin v1.2.3b7
}

cut() {
  cd "${WORK}" && bash "${REPO_ROOT}/scripts/release.sh"
}

promote() {
  cd "${WORK}" && bash "${REPO_ROOT}/scripts/promote_release.sh" v1.2.3b7 1.2.3b8
}

@test "cutting a release under tag.gpgsign needs no editor" {
  make_repo true
  run cut
  [ "$status" -eq 0 ]
  [ ! -s "${EDITOR_LOG}" ]
  [ "$(git -C "${WORK}" tag -l)" = "v1.2.3b8" ]
  [ "$(git -C "${WORK}" cat-file -t v1.2.3b8)" = "commit" ]
  [ "$(git -C "${ORIGIN}" cat-file -t v1.2.3b8)" = "commit" ]
}

@test "cutting a release without tag signing behaves the same" {
  make_repo false
  run cut
  [ "$status" -eq 0 ]
  [ ! -s "${EDITOR_LOG}" ]
  [ "$(git -C "${WORK}" tag -l)" = "v1.2.3b8" ]
  [ "$(git -C "${WORK}" cat-file -t v1.2.3b8)" = "commit" ]
  [ "$(git -C "${ORIGIN}" cat-file -t v1.2.3b8)" = "commit" ]
}

@test "promoting a tag under tag.gpgsign needs no editor" {
  make_repo true
  seed_source_tag
  run promote
  [ "$status" -eq 0 ]
  [ ! -s "${EDITOR_LOG}" ]
  [ "$(git -C "${ORIGIN}" tag -l v1.2.3b8)" = "v1.2.3b8" ]
  [ "$(git -C "${ORIGIN}" cat-file -t v1.2.3b8)" = "commit" ]
}

@test "promoting a tag without tag signing behaves the same" {
  make_repo false
  seed_source_tag
  run promote
  [ "$status" -eq 0 ]
  [ ! -s "${EDITOR_LOG}" ]
  [ "$(git -C "${ORIGIN}" tag -l v1.2.3b8)" = "v1.2.3b8" ]
  [ "$(git -C "${ORIGIN}" cat-file -t v1.2.3b8)" = "commit" ]
}
