#!/usr/bin/env bats
# tools/ci/pick_engine_tag.sh against a tag list and the real denylist.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/tools/ci/pick_engine_tag.sh"
  DENYLIST="${REPO_ROOT}/engine-bad-tags.list"
}

pick() {
  printf '%s\n' "$@" | bash "${SCRIPT}" "${DENYLIST}"
}

@test "the newest tag wins when it is not listed" {
  run pick memory-20260914-080000 memory-20261102-180518
  [ "$status" -eq 0 ]
  [ "$output" = "memory-20261102-180518" ]
}

@test "a listed bad tag that is the newest is skipped with its reason" {
  run pick memory-20260914-080000 memory-20260921-150014
  [ "$status" -eq 0 ]
  [[ "$output" == *"skipping memory-20260921-150014:"*"fattn.cu"* ]]
  [[ "$output" == *"memory-20260914-080000" ]]
}

@test "the engine pin file names the denylist, not a second record" {
  run grep -F 'engine-bad-tags.list' "${REPO_ROOT}/engine-versions.env"
  [ "$status" -eq 0 ]
  run grep -F 'Do not move this pin to' "${REPO_ROOT}/engine-versions.env"
  [ "$status" -ne 0 ]
}

@test "the workflow selects the tag through the script" {
  run grep -F 'tools/ci/pick_engine_tag.sh engine-bad-tags.list' "${REPO_ROOT}/.github/workflows/engine-pin-sync.yml"
  [ "$status" -eq 0 ]
}

@test "the run fails when only bad tags exist" {
  run pick memory-20260921-150014
  [ "$status" -eq 1 ]
}
