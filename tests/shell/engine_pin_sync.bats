#!/usr/bin/env bats
# .github/workflows/engine-pin-sync.yml: the rolling pull request must get CI.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  WORKFLOW="${REPO_ROOT}/.github/workflows/engine-pin-sync.yml"
}

# The run script of the step that opens or refreshes the pull request.
pr_step() {
  yq -r '.jobs.bump.steps[] | select(.name == "Open or refresh the rolling PR") | .run' "${WORKFLOW}"
}

@test "the step that opens the pull request is found" {
  run pr_step
  [ "$status" -eq 0 ]
  [[ "$output" == *"gh pr create"* ]]
}

@test "a new pull request gets a second push with the deploy key" {
  run pr_step
  [ "$status" -eq 0 ]
  # Lines of the create branch: after `gh pr create`, before the `else`.
  created=$(printf '%s\n' "$output" | awk '/gh pr create/ { on = 1 } /^ *else/ { on = 0 } on')
  [ -n "${created}" ]
  [[ "${created}" == *"git commit -q --allow-empty"* ]]
  [[ "${created}" == *"git push origin chore/engine-pin-sync"* ]]
}

@test "the checkout carries the deploy key that the pushes use" {
  run yq -r '.jobs.bump.steps[0].with["ssh-key"]' "${WORKFLOW}"
  [ "$output" = '${{ secrets.ENGINE_PIN_SYNC_KEY }}' ]
}
