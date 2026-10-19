#!/usr/bin/env bats
# scripts/flake_pin_pr.sh lands a flake pin through a pull request to main,
# never a push to main. Real git against a local bare remote; gh is stubbed.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/flake_pin_pr.sh"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  mkdir -p "${FIXTURE}/bin"
  cp "${BATS_TEST_DIRNAME}/stubs/gh-flake-pin" "${FIXTURE}/bin/gh"
  chmod +x "${FIXTURE}/bin/gh"
  : > "${FIXTURE}/gh.log"

  REMOTE="${BATS_TEST_TMPDIR}/remote.git"
  WORK="${BATS_TEST_TMPDIR}/work"
  git init -q --bare -b main "${REMOTE}"
  git init -q -b main "${WORK}"
  git -C "${WORK}" config user.name test
  git -C "${WORK}" config user.email test@example.com
  for f in sources.json sources-cuda.json; do
    echo '{"version":"1.0.0"}' > "${WORK}/${f}"
  done
  git -C "${WORK}" add .
  git -C "${WORK}" commit -q -m "Initial"
  git -C "${WORK}" remote add origin "${REMOTE}"
  git -C "${WORK}" push -q origin main
}

# pin <target file> <version> [dry run]
pin() {
  echo "{\"version\":\"$2\"}" > "${WORK}/$1"
  ( cd "${WORK}" && FIXTURE="${FIXTURE}" PATH="${FIXTURE}/bin:${PATH}" \
      TARGET_FILE="$1" VERSION="$2" COMMIT_MESSAGE="pin $1 $2" DRY_RUN="${3:-false}" \
      bash "${SCRIPT}" )
}

remote_branches() {
  git -C "${REMOTE}" for-each-ref --format='%(refname:short)' refs/heads | sort | paste -sd, -
}

@test "the default sources file pins on the lilbee branch" {
  run pin sources.json 2.0.0
  [ "$status" -eq 0 ]
  [ "$(remote_branches)" = "flake-pins/lilbee-2.0.0,main" ]
}

@test "a variant sources file pins on its own package branch" {
  run pin sources-cuda.json 2.0.0
  [ "$status" -eq 0 ]
  [ "$(remote_branches)" = "flake-pins/lilbee-cuda-2.0.0,main" ]
}

@test "a file that is not a sources file is refused" {
  run pin flake.lock 2.0.0
  [ "$status" -eq 2 ]
  [ "$(remote_branches)" = "main" ]
}

@test "main is never pushed" {
  before=$(git -C "${REMOTE}" rev-parse main)
  run pin sources.json 2.0.0
  [ "$status" -eq 0 ]
  [ "$(git -C "${REMOTE}" rev-parse main)" = "${before}" ]
}

@test "the pin branch carries the new sources file" {
  run pin sources.json 2.0.0
  [ "$status" -eq 0 ]
  run git -C "${REMOTE}" show flake-pins/lilbee-2.0.0:sources.json
  [ "$output" = '{"version":"2.0.0"}' ]
}

@test "a new pin opens a pull request, then pushes again so its CI starts, then auto-merges" {
  run pin sources.json 2.0.0
  [ "$status" -eq 0 ]
  grep -q '^pr create --base main --head flake-pins/lilbee-2.0.0 --title pin sources.json 2.0.0 ' "${FIXTURE}/gh.log"
  grep -qx 'pr merge flake-pins/lilbee-2.0.0 --auto --squash' "${FIXTURE}/gh.log"
  # The second commit is the deploy-key push that the pull request's CI runs on.
  [ "$(git -C "${REMOTE}" rev-list --count main..flake-pins/lilbee-2.0.0)" -eq 2 ]
}

@test "an open pull request for the version is updated, not opened again" {
  echo '[{"number":41,"headRefName":"flake-pins/lilbee-2.0.0"}]' > "${FIXTURE}/open_prs.json"
  run pin sources.json 2.0.0
  [ "$status" -eq 0 ]
  run grep -c '^pr create' "${FIXTURE}/gh.log"
  [ "$output" = "0" ]
  grep -qx 'pr merge flake-pins/lilbee-2.0.0 --auto --squash' "${FIXTURE}/gh.log"
  [ "$(git -C "${REMOTE}" rev-list --count main..flake-pins/lilbee-2.0.0)" -eq 1 ]
}

@test "a rerun rebuilds the branch on the current main" {
  run pin sources.json 2.0.0
  echo '[{"number":41,"headRefName":"flake-pins/lilbee-2.0.0"}]' > "${FIXTURE}/open_prs.json"
  echo moved > "${WORK}/other"
  git -C "${WORK}" checkout -q main
  git -C "${WORK}" add other
  git -C "${WORK}" commit -q -m "Move main"
  git -C "${WORK}" push -q origin main
  run pin sources.json 2.0.0
  [ "$status" -eq 0 ]
  [ "$(git -C "${REMOTE}" merge-base main flake-pins/lilbee-2.0.0)" = "$(git -C "${REMOTE}" rev-parse main)" ]
}

@test "an older open pin pull request for the same package is closed" {
  echo '[{"number":40,"headRefName":"flake-pins/lilbee-1.9.0"},{"number":39,"headRefName":"flake-pins/lilbee-cuda-1.9.0"}]' \
    > "${FIXTURE}/open_prs.json"
  run pin sources.json 2.0.0
  [ "$status" -eq 0 ]
  grep -q '^pr close 40 ' "${FIXTURE}/gh.log"
  run grep -c '^pr close 39' "${FIXTURE}/gh.log"
  [ "$output" = "0" ]
}

@test "a sources file that did not change opens nothing" {
  run pin sources.json 1.0.0
  [ "$status" -eq 0 ]
  [ "$(remote_branches)" = "main" ]
  [ ! -s "${FIXTURE}/gh.log" ]
}

@test "a dry run names the branch, commit and pull request and pushes nothing" {
  run pin sources.json 2.0.0 true
  [ "$status" -eq 0 ]
  [[ "$output" == *"branch: flake-pins/lilbee-2.0.0"* ]]
  [[ "$output" == *"commit: "*"pin sources.json 2.0.0"* ]]
  [[ "$output" == *"would open a pull request from flake-pins/lilbee-2.0.0 to main"* ]]
  [ "$(remote_branches)" = "main" ]
  run grep -c -E '^pr (create|merge|close)' "${FIXTURE}/gh.log"
  [ "$output" = "0" ]
}

@test "a dry run names the open pull request it would update" {
  echo '[{"number":41,"headRefName":"flake-pins/lilbee-2.0.0"}]' > "${FIXTURE}/open_prs.json"
  run pin sources.json 2.0.0 true
  [ "$status" -eq 0 ]
  [[ "$output" == *"would update pull request #41"* ]]
}

@test "the publish action lands the pin through this script" {
  run grep -c 'scripts/flake_pin_pr.sh' "${REPO_ROOT}/.github/actions/publish-flake-sources/action.yml"
  [ "$output" = "1" ]
  run grep -c -E 'git push|git_push_retry' "${REPO_ROOT}/.github/actions/publish-flake-sources/action.yml"
  [ "$output" = "0" ]
}

@test "every nix-flake job checks out with the flake pins key" {
  for wf in publish-packages publish-cuda-packages publish-rocm-packages publish-compat-packages; do
    run yq -r '.jobs["nix-flake"].steps[0].with["ssh-key"]' "${REPO_ROOT}/.github/workflows/${wf}.yml"
    [ "$output" = '${{ secrets.FLAKE_PINS_SSH_KEY }}' ] || { echo "${wf}: ${output}" >&2; return 1; }
  done
}

@test "a dry run of publish packages stops before any channel unless it is nix alone" {
  wf="${REPO_ROOT}/.github/workflows/publish-packages.yml"
  run yq -r '.jobs.resolve.steps[0].if' "${wf}"
  [ "$output" = "inputs.dry_run" ]
  run yq -r '.jobs.resolve.steps[0].run' "${wf}"
  [[ "$output" == *'[ "${CHANNELS}" = "nix" ]'* ]]
  # Every channel waits for resolve, so the guard stops all of them.
  run yq -r '.jobs | to_entries | .[] | select(.key != "resolve") | .key + "=" + .value.needs' "${wf}"
  [ "$status" -eq 0 ]
  [ -n "$output" ]
  run grep -v -x -E '[a-z-]+=resolve' <<< "$output"
  [ "$output" = "" ]
}
