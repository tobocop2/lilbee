#!/usr/bin/env bats
# Third-party actions that install a toolchain are pinned by commit, not by a tag a maintainer can move.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
}

@test "the nix installer action is pinned by commit everywhere" {
  run grep -rhoE 'uses: DeterminateSystems/nix-installer-action@[^ ]+' \
    "${REPO_ROOT}/.github/workflows" "${REPO_ROOT}/.github/actions"
  [ "$status" -eq 0 ]
  [ -n "$output" ]
  bad=$(printf '%s\n' "$output" | grep -v -E '@[0-9a-f]{40}$' || true)
  [ -z "${bad}" ] || { echo "not pinned by commit: ${bad}" >&2; return 1; }
}

@test "every use of the nix installer action pins the same commit" {
  run bash -c "grep -rhoE 'nix-installer-action@[^ ]+' '${REPO_ROOT}/.github' | sort -u | wc -l"
  [ "$(printf '%s' "$output" | tr -d ' ')" = "1" ]
}
