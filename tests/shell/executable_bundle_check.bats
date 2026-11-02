#!/usr/bin/env bats
# Every workflow that builds an executable runs the engine bundle check on the
# Nuitka tree after the build and before the step that deletes that tree.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
}

# One line per step: "<index> <kind>", kind is build, check or reclaim, else other.
step_kinds() {
  yq -r '.jobs.build.steps | to_entries | .[] | (.key | tostring) + " " + (if ((.value.run // "") | contains("build_lilbee_binary.sh")) then "build" elif ((.value.run // "") | contains("assert_engine_bundle.py")) then "check" elif ((.value.run // "") | contains("dist/*.dist")) then "reclaim" else "other" end)' \
    "${REPO_ROOT}/.github/workflows/$1"
}

index_of() {
  step_kinds "$1" | awk -v kind="$2" '$2 == kind { print $1 }'
}

assert_check_between_build_and_reclaim() {
  local build check reclaim
  build="$(index_of "$1" build)"
  check="$(index_of "$1" check)"
  reclaim="$(index_of "$1" reclaim)"
  [ -n "${build}" ]
  [ -n "${check}" ]
  [ "$(printf '%s\n' "${check}" | wc -l)" -eq 1 ]
  [ "${check}" -gt "${build}" ]
  if [ -n "${reclaim}" ]; then
    [ "${check}" -lt "${reclaim}" ]
  fi
}

@test "release.yml checks the engine bundle after the build" {
  assert_check_between_build_and_reclaim release.yml
}

@test "release.yml still reclaims the build tree, after the check" {
  [ -n "$(index_of release.yml reclaim)" ]
}

@test "build-gpu-executables.yml checks the engine bundle after the build" {
  assert_check_between_build_and_reclaim build-gpu-executables.yml
}

@test "the check step has no condition and cannot fail softly" {
  for wf in release.yml build-gpu-executables.yml; do
    idx="$(index_of "${wf}" check)"
    run yq -r ".jobs.build.steps[${idx}] | [.[\"if\"], .[\"continue-on-error\"]] | map(. == null) | all" \
      "${REPO_ROOT}/.github/workflows/${wf}"
    [ "$status" -eq 0 ]
    [ "$output" = "true" ]
  done
}

@test "build-gpu-executables.yml keeps no second copy of the runtime list" {
  run grep -c 'compgen -G' "${REPO_ROOT}/.github/workflows/build-gpu-executables.yml"
  [ "$output" = "0" ]
}

@test "the check passes a runner OS and a bin directory" {
  for wf in release.yml build-gpu-executables.yml; do
    run grep -F -- '--backend "${BACKEND}" --platform "${RUNNER_OS}" "${bin_dir}"' \
      "${REPO_ROOT}/.github/workflows/${wf}"
    [ "$status" -eq 0 ]
  done
}
