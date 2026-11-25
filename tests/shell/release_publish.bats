#!/usr/bin/env bats
# scripts/resolve_rc_run.sh and its call sites against the stubbed gh.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/resolve_rc_run.sh"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  mkdir -p "${FIXTURE}/runs" "${FIXTURE}/views" "${FIXTURE}/bin"
  : > "${FIXTURE}/actions.log"
  cp "${BATS_TEST_DIRNAME}/stubs/gh" "${FIXTURE}/bin/gh"
  chmod +x "${FIXTURE}/bin/gh"
}

resolve() {  # RC_RUN_ID may be set by the caller
  FIXTURE="${FIXTURE}" GH_TOKEN=stub REPO=tobocop2/lilbee TAG=v1.0.0 SHA=good \
  PATH="${FIXTURE}/bin:${PATH}" bash "${SCRIPT}"
}

runs() {  # the release-candidate list the stub serves
  printf '%s\n' "$1" > "${FIXTURE}/runs/release-candidate.yml.json"
}

@test "the newest run by creation time wins" {
  runs '[{"databaseId":2,"createdAt":"2026-09-21T23:38:52Z","conclusion":"success","headSha":"good"},
         {"databaseId":1,"createdAt":"2026-09-21T22:05:08Z","conclusion":"success","headSha":"good"}]'
  run resolve
  [ "$status" -eq 0 ]
  [ "$output" = "2" ]
}

@test "a run of another commit loses even when it is newest" {
  runs '[{"databaseId":1,"createdAt":"2026-09-21T22:05:08Z","conclusion":"success","headSha":"good"},
         {"databaseId":2,"createdAt":"2026-09-21T23:38:52Z","conclusion":"success","headSha":"stale"}]'
  run resolve
  [ "$status" -eq 0 ]
  [ "$output" = "1" ]
}

@test "a cancelled run is never picked" {
  runs '[{"databaseId":1,"createdAt":"2026-09-21T22:05:08Z","conclusion":"success","headSha":"good"},
         {"databaseId":2,"createdAt":"2026-09-21T23:38:52Z","conclusion":"cancelled","headSha":"good"}]'
  run resolve
  [ "$status" -eq 0 ]
  [ "$output" = "1" ]
}

@test "no run of the tag commit is a refusal" {
  runs '[{"databaseId":2,"createdAt":"2026-09-21T23:38:52Z","conclusion":"success","headSha":"stale"}]'
  run resolve
  [ "$status" -eq 1 ]
  [[ "$output" == *"no uncancelled 'Release candidate' run"* ]]
}

@test "an explicit run that built another commit is refused" {
  echo '{"headSha":"stale","conclusion":"success"}' > "${FIXTURE}/views/7.json"
  RC_RUN_ID=7 run resolve
  [ "$status" -eq 1 ]
  [[ "$output" == *"refusing to publish a different commit"* ]]
}

@test "an explicit cancelled run is refused" {
  echo '{"headSha":"good","conclusion":"cancelled"}' > "${FIXTURE}/views/7.json"
  RC_RUN_ID=7 run resolve
  [ "$status" -eq 1 ]
  [[ "$output" == *"was cancelled"* ]]
}

@test "an explicit run of the tag commit is used" {
  echo '{"headSha":"good","conclusion":"failure"}' > "${FIXTURE}/views/7.json"
  RC_RUN_ID=7 run resolve
  [ "$status" -eq 0 ]
  [ "$output" = "7" ]
}

@test "publish.yml resolves the run through the script" {
  run yq -r '.jobs.guard.steps[] | select(.id == "resolve") | .run' "${REPO_ROOT}/.github/workflows/publish.yml"
  [ "$status" -eq 0 ]
  [[ "$output" == *"scripts/resolve_rc_run.sh"* ]]
  [[ "$output" != *"run_started_at"* ]]
}

@test "publish.yml executes the resolver and publishes the run id it prints" {
  # Run the real step body in a scratch checkout. The resolver there is a
  # sentinel, so the output proves the step ran it and took its answer.
  work="${BATS_TEST_TMPDIR}/work"
  mkdir -p "${work}/scripts" "${work}/bin"
  printf '#!/usr/bin/env bash\necho ran >> "%s/resolver.log"\necho 4242\n' "${BATS_TEST_TMPDIR}" > "${work}/scripts/resolve_rc_run.sh"
  printf '#!/usr/bin/env bash\necho good\n' > "${work}/bin/git"
  printf '#!/usr/bin/env bash\necho "{\\"artifacts\\":[{\\"name\\":\\"wheel-default\\"},{\\"name\\":\\"sdist\\"}]}"\n' > "${work}/bin/gh"
  chmod +x "${work}/bin/git" "${work}/bin/gh"
  yq -r '.jobs.guard.steps[] | select(.id == "resolve") | .run' "${REPO_ROOT}/.github/workflows/publish.yml" > "${work}/step.sh"
  : > "${work}/out"
  cd "${work}"
  GITHUB_OUTPUT="${work}/out" REPO=tobocop2/lilbee TAG=v1.0.0 PATH="${work}/bin:${PATH}" run bash step.sh
  [ "$status" -eq 0 ]
  [ "$(cat "${BATS_TEST_TMPDIR}/resolver.log")" = "ran" ]
  [ "$(cat "${work}/out")" = "rc_run_id=4242" ]
}

@test "the release candidate hands publish.yml its own run id" {
  run yq -r '.jobs["dispatch-publish"].steps[0].run' "${REPO_ROOT}/.github/workflows/release-candidate.yml"
  [ "$status" -eq 0 ]
  [[ "$output" == *'rc_run_id="${RUN_ID}"'* ]]
  run yq -r '.jobs["dispatch-publish"].steps[0].env.RUN_ID' "${REPO_ROOT}/.github/workflows/release-candidate.yml"
  [ "$output" = '${{ github.run_id }}' ]
}

@test "emergency-publish orders runs by creation time too" {
  run grep -c 'sort_by(.created_at)' "${REPO_ROOT}/.github/workflows/emergency-publish.yml"
  [ "$output" = "1" ]
  run grep -c 'run_started_at' "${REPO_ROOT}/.github/workflows/emergency-publish.yml"
  [ "$output" = "0" ]
}
