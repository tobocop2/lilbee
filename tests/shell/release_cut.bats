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

npm_files() {  # an npm package whose version is stale, as a hand-bumped one is
  printf '{\n  "name": "lilbee",\n  "version": "9.9.9",\n  "lilbee": {\n    "repo": "o/r",\n    "release": "v0.0.1",\n    "glibcFloor": "2.34"\n  }\n}\n' \
    > "${WORK}/packages/lilbee/package.json"
  printf '{\n  "name": "lilbee",\n  "version": "9.9.9",\n  "lockfileVersion": 3,\n  "packages": {\n    "": {\n      "name": "lilbee",\n      "version": "9.9.9",\n      "license": "MIT"\n    },\n    "node_modules/undici": {\n      "version": "9.9.9"\n    }\n  }\n}\n' \
    > "${WORK}/packages/lilbee/package-lock.json"
}

make_repo() {  # tag.gpgsign value, then the seed version (default 1.2.3b7)
  local seed="${2:-1.2.3b7}"
  git init -q --bare "${ORIGIN}"
  git init -q -b main "${WORK}"
  git -C "${WORK}" config user.name "Release Test"
  git -C "${WORK}" config user.email "release@example.com"
  git -C "${WORK}" config commit.gpgsign false
  git -C "${WORK}" config tag.gpgsign "$1"
  printf 'version = "%s"\n' "${seed}" > "${WORK}/pyproject.toml"
  printf 'version = "%s"\n' "${seed}" > "${WORK}/uv.lock"
  mkdir -p "${WORK}/plugins/lilbee/.claude-plugin"
  printf '{\n  "name": "lilbee",\n  "version": "%s",\n  "license": "MIT"\n}\n' "${seed}" > "${WORK}/plugins/lilbee/.claude-plugin/plugin.json"
  mkdir -p "${WORK}/packages/lilbee"
  npm_files "${seed}"
  git -C "${WORK}" add pyproject.toml uv.lock plugins/lilbee/.claude-plugin/plugin.json packages/lilbee/package.json packages/lilbee/package-lock.json
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

@test "cutting a release moves the plugin version with the lilbee version" {
  make_repo false
  run cut
  [ "$status" -eq 0 ]
  grep -q '"version": "1.2.3b8"' "${WORK}/plugins/lilbee/.claude-plugin/plugin.json"
  [ -z "$(git -C "${WORK}" status --porcelain)" ]
  run git -C "${WORK}" show --stat --format= HEAD
  [[ "$output" == *"plugins/lilbee/.claude-plugin/plugin.json"* ]]
}

@test "a release stops when the plugin version does not follow the lilbee version" {
  make_repo false
  printf '{\n  "version": "9.9.9"\n}\n' > "${WORK}/plugins/lilbee/.claude-plugin/plugin.json"
  git -C "${WORK}" commit -q -a -m "Drift"
  git -C "${WORK}" push -q origin main
  run cut
  [ "$status" -eq 1 ]
  [[ "$output" == *"plugin.json did not take version 1.2.3b8"* ]]
  [ -z "$(git -C "${WORK}" status --porcelain)" ]
  [ -z "$(git -C "${WORK}" tag -l)" ]
}

detect() {  # tag: run detect_promotion.sh at the checked-out release commit
  cd "${WORK}" && bash "${REPO_ROOT}/scripts/detect_promotion.sh" "$1"
}

@test "a dev counter moves the plugin version to the next dev counter" {
  make_repo false 1.2.3b7.dev4
  run cut
  [ "$status" -eq 0 ]
  grep -q '"version": "1.2.3b7.dev5"' "${WORK}/plugins/lilbee/.claude-plugin/plugin.json"
}

@test "two releases in a row are not a promotion" {
  make_repo false
  run cut
  [ "$status" -eq 0 ]
  run cut
  [ "$status" -eq 0 ]
  [ "$(git -C "${WORK}" rev-parse HEAD^)" = "$(git -C "${WORK}" rev-parse v1.2.3b8^{commit})" ]
  run detect v1.2.3b9
  [ "$status" -eq 0 ]
  [ "$output" = "promoted_from=" ]
}

@test "a promoted tag is detected as a promotion of its source tag" {
  make_repo false
  seed_source_tag
  run promote
  [ "$status" -eq 0 ]
  git -C "${WORK}" fetch -q origin tag v1.2.3b8
  git -C "${WORK}" checkout -q --detach v1.2.3b8
  run detect v1.2.3b8
  [ "$status" -eq 0 ]
  [[ "$output" == *"promoted_from=v1.2.3b7" ]]
}

@test "a promotion moves the plugin manifest to the new version" {
  make_repo false
  seed_source_tag
  run promote
  [ "$status" -eq 0 ]
  git -C "${WORK}" fetch -q origin tag v1.2.3b8
  [ "$(git -C "${WORK}" show v1.2.3b8:pyproject.toml)" = 'version = "1.2.3b8"' ]
  git -C "${WORK}" show v1.2.3b8:plugins/lilbee/.claude-plugin/plugin.json | grep -q '"version": "1.2.3b8"'
}

@test "promoting a dev tag to its beta gives the plugin the beta version" {
  make_repo false 1.2.3b7.dev4
  git -C "${WORK}" -c tag.gpgsign=false tag v1.2.3b7.dev4
  git -C "${WORK}" push -q origin v1.2.3b7.dev4
  cd "${WORK}"
  run bash "${REPO_ROOT}/scripts/promote_release.sh" v1.2.3b7.dev4 1.2.3b7
  [ "$status" -eq 0 ]
  git -C "${WORK}" fetch -q origin tag v1.2.3b7
  git -C "${WORK}" show v1.2.3b7:plugins/lilbee/.claude-plugin/plugin.json | grep -q '"version": "1.2.3b7"'
}

@test "a promotion of a tag from before the plugin and npm package existed changes only the Python version files" {
  make_repo false
  git -C "${WORK}" rm -q -r plugins packages
  git -C "${WORK}" commit -q -m "No plugin"
  git -C "${WORK}" push -q origin main
  seed_source_tag
  run promote
  [ "$status" -eq 0 ]
  git -C "${WORK}" fetch -q origin tag v1.2.3b8
  [ "$(git -C "${WORK}" diff --name-only v1.2.3b7 v1.2.3b8 | tr '\n' ' ')" = "pyproject.toml uv.lock " ]
}

@test "cutting a release stamps the npm version, the release tag and the lock file" {
  make_repo false 1.2.3b999
  run cut
  [ "$status" -eq 0 ]
  grep -q '^  "version": "1.2.3-beta.1000",$' "${WORK}/packages/lilbee/package.json"
  grep -q '^    "release": "v1.2.3b1000",$' "${WORK}/packages/lilbee/package.json"
  [ "$(head -5 "${WORK}/packages/lilbee/package-lock.json" | grep -c '"version": "1.2.3-beta.1000"')" = 1 ]
  grep -c '"version": "1.2.3-beta.1000"' "${WORK}/packages/lilbee/package-lock.json" | grep -q '^2$'
  grep -q '"version": "9.9.9"' "${WORK}/packages/lilbee/package-lock.json"
  [ -z "$(git -C "${WORK}" status --porcelain)" ]
  run git -C "${WORK}" show --stat --format= HEAD
  [[ "$output" == *"packages/lilbee/package.json"* ]]
  [[ "$output" == *"packages/lilbee/package-lock.json"* ]]
  [[ "$output" == *"plugins/lilbee/.claude-plugin/plugin.json"* ]]
}

@test "a release stops, with nothing changed, when the npm package does not take the version" {
  make_repo false
  printf '{\n  "name": "lilbee"\n}\n' > "${WORK}/packages/lilbee/package.json"
  git -C "${WORK}" commit -q -a -m "Drift"
  git -C "${WORK}" push -q origin main
  run cut
  [ "$status" -eq 1 ]
  [[ "$output" == *"package.json did not take version 1.2.3-beta.8"* ]]
  [ -z "$(git -C "${WORK}" status --porcelain)" ]
  [ -z "$(git -C "${WORK}" tag -l)" ]
}

@test "a dev counter stamps the npm package with the dev form" {
  make_repo false 1.2.3b7.dev4
  run cut
  [ "$status" -eq 0 ]
  grep -q '^  "version": "1.2.3-beta.7.dev.5",$' "${WORK}/packages/lilbee/package.json"
  grep -q '^    "release": "v1.2.3b7.dev5",$' "${WORK}/packages/lilbee/package.json"
}

@test "a promotion moves the npm package to the new version and is still detected" {
  make_repo false
  seed_source_tag
  run promote
  [ "$status" -eq 0 ]
  git -C "${WORK}" fetch -q origin tag v1.2.3b8
  git -C "${WORK}" show v1.2.3b8:packages/lilbee/package.json | grep -q '^  "version": "1.2.3-beta.8",$'
  git -C "${WORK}" show v1.2.3b8:packages/lilbee/package.json | grep -q '^    "release": "v1.2.3b8",$'
  git -C "${WORK}" checkout -q --detach v1.2.3b8
  run detect v1.2.3b8
  [ "$status" -eq 0 ]
  [[ "$output" == *"promoted_from=v1.2.3b7" ]]
}

@test "a promotion to a version with no npm mapping stops before it commits" {
  make_repo false
  seed_source_tag
  cd "${WORK}"
  run bash "${REPO_ROOT}/scripts/promote_release.sh" v1.2.3b7 1.2.3rc1
  [ "$status" -ne 0 ]
  [[ "$output" == *"'1.2.3rc1' has no npm version"* ]]
  [ -z "$(git -C "${ORIGIN}" tag -l v1.2.3rc1)" ]
}
