#!/usr/bin/env bats
# .github/workflows/engine-pin-sync.yml: the rolling pull request must get CI.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  WORKFLOW="${REPO_ROOT}/.github/workflows/engine-pin-sync.yml"
}

# The run script of the named step, without comment lines: a commented-out
# command must not satisfy a test.
step_run() {
  yq -r --arg name "$1" '.jobs.bump.steps[] | select(.name == $name) | .run' "${WORKFLOW}" \
    | grep -v -E '^[[:space:]]*#'
}

@test "the step that opens the pull request runs gh pr create" {
  run step_run "Open or refresh the rolling PR"
  [ "$status" -eq 0 ]
  printf '%s\n' "$output" | grep -q -E '^[[:space:]]*gh pr create'
}

@test "a new pull request gets a second push with the deploy key" {
  run step_run "Open or refresh the rolling PR"
  [ "$status" -eq 0 ]
  # Lines of the create branch: after `gh pr create`, before the `else`.
  created=$(printf '%s\n' "$output" | awk '/^[[:space:]]*gh pr create/ { on = 1 } /^[[:space:]]*else/ { on = 0 } on')
  [ -n "${created}" ]
  printf '%s\n' "${created}" | grep -q -E '^[[:space:]]*git commit -q --allow-empty'
  printf '%s\n' "${created}" | grep -q -E '^[[:space:]]*bash scripts/ci_retry\.sh 3 git push origin chore/engine-pin-sync$'
}

@test "the branch push retries a transient failure" {
  run step_run "Push the rolling branch"
  [ "$status" -eq 0 ]
  printf '%s\n' "$output" | grep -q -E '^[[:space:]]*bash scripts/ci_retry\.sh 3 git push -f origin chore/engine-pin-sync$'
}

@test "the checkout carries the deploy key that the pushes use" {
  run yq -r '.jobs.bump.steps[0].with["ssh-key"]' "${WORKFLOW}"
  [ "$output" = '${{ secrets.ENGINE_PIN_SYNC_KEY }}' ]
}
