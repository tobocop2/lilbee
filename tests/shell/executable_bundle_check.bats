#!/usr/bin/env bats
# Every workflow that builds an executable runs the engine bundle check on the
# Nuitka tree after the build and before the step that deletes or uploads it.
#
# The workflows are read as YAML with yq, and a step's script is read as code:
# a line that starts with # is a comment, so a commented-out call does not count.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  WORKFLOWS="release.yml build-gpu-executables.yml flatpak-validate.yml"
}

# The code lines of step <index> of workflow <file>: every line whose first
# non-blank character is # is dropped. Plain yq paths only: the laptop's yq is
# the jq wrapper, CI's is the Go one (mikefarah v4), and both read these.
step_code() {
  yq -r ".jobs.build.steps[$2].run // \"\"" \
    "${REPO_ROOT}/.github/workflows/$1" | awk '!/^[[:space:]]*#/'
}

# One line per step: "<index> <kind>", kind is build, check, reclaim or upload, else other.
step_kinds() {
  local file="${REPO_ROOT}/.github/workflows/$1" count i code uses kind
  count="$(yq -r '.jobs.build.steps | length' "${file}")"
  for ((i = 0; i < count; i++)); do
    code="$(step_code "$1" "${i}")"
    uses="$(yq -r ".jobs.build.steps[${i}].uses // \"\"" "${file}")"
    if [[ "${code}" == *"build_lilbee_binary.sh"* ]]; then
      kind=build
    elif [[ "${code}" == *"assert_engine_bundle.py"* ]]; then
      kind=check
    elif [[ "${code}" == *"dist/*.dist"* ]]; then
      kind=reclaim
    elif [[ "${uses}" == actions/upload-artifact* ]]; then
      kind=upload
    else
      kind=other
    fi
    echo "${i} ${kind}"
  done
}

index_of() {
  step_kinds "$1" | awk -v kind="$2" '$2 == kind { print $1 }'
}

# Succeeds when workflow <file> has exactly one check step, after the build and
# before every step that deletes or uploads the tree.
assert_check_between_build_and_after() {
  local build check later
  build="$(index_of "$1" build)"
  check="$(index_of "$1" check)"
  [ -n "${build}" ]
  [ -n "${check}" ]
  [ "$(printf '%s\n' "${check}" | wc -l)" -eq 1 ]
  [ "${check}" -gt "${build}" ]
  for later in $(index_of "$1" reclaim) $(index_of "$1" upload); do
    [ "${check}" -lt "${later}" ]
  done
}

# Run the check step of workflow <file> in a scratch directory that holds the
# trees named after it (paths under dist/; none means no dist at all). A stub
# uv records the checker command line in uv.args.
run_check_step() {
  local wf="$1" idx tree
  shift
  idx="$(index_of "${wf}" check)"
  WORK="$(mktemp -d "${BATS_TEST_TMPDIR}/step.XXXXXX")"
  mkdir -p "${WORK}/stub"
  ln -s "${REPO_ROOT}/tools" "${WORK}/tools"
  printf '#!/bin/sh\necho "$@" > "%s/uv.args"\n' "${WORK}" > "${WORK}/stub/uv"
  chmod +x "${WORK}/stub/uv"
  for tree in "$@"; do
    mkdir -p "${WORK}/${tree}"
  done
  step_code "${wf}" "${idx}" > "${WORK}/step.sh"
  cd "${WORK}"
  run env PATH="${WORK}/stub:${PATH}" BACKEND=vulkan RUNNER_OS=Linux \
    GITHUB_ENV="${WORK}/ghenv" bash ./step.sh </dev/null
}

@test "release.yml checks the engine bundle after the build" {
  assert_check_between_build_and_after release.yml
}

@test "release.yml still reclaims the build tree, after the check" {
  [ -n "$(index_of release.yml reclaim)" ]
  [ "$(index_of release.yml check)" -lt "$(index_of release.yml reclaim)" ]
}

@test "build-gpu-executables.yml checks the engine bundle after the build" {
  assert_check_between_build_and_after build-gpu-executables.yml
}

@test "flatpak-validate.yml checks the engine bundle before the upload" {
  assert_check_between_build_and_after flatpak-validate.yml
  [ -n "$(index_of flatpak-validate.yml upload)" ]
}

@test "the check step has no condition and cannot fail softly" {
  for wf in ${WORKFLOWS}; do
    idx="$(index_of "${wf}" check)"
    run yq -r ".jobs.build.steps[${idx}] | has(\"if\")" "${REPO_ROOT}/.github/workflows/${wf}"
    [ "$status" -eq 0 ]
    [ "$output" = "false" ]
    run yq -r ".jobs.build.steps[${idx}] | has(\"continue-on-error\")" "${REPO_ROOT}/.github/workflows/${wf}"
    [ "$status" -eq 0 ]
    [ "$output" = "false" ]
  done
}

@test "build-gpu-executables.yml keeps no second copy of the runtime list" {
  run grep -c 'compgen -G' "${REPO_ROOT}/.github/workflows/build-gpu-executables.yml"
  [ "$output" = "0" ]
}

@test "the check step stops on a failure, finds the tree and passes a runner OS" {
  for wf in ${WORKFLOWS}; do
    code="$(step_code "${wf}" "$(index_of "${wf}" check)")"
    printf '%s\n' "${code}" | grep -qF 'set -euo pipefail'
    printf '%s\n' "${code}" | grep -qF 'bin_dir=$(bash tools/qa/find_engine_bin.sh dist)'
    printf '%s\n' "${code}" | grep -qF -- '--platform "${RUNNER_OS}" "${bin_dir}"'
  done
}

@test "every check step fails when the build left no engine tree, or no dist" {
  for wf in ${WORKFLOWS}; do
    run_check_step "${wf}"
    [ "$status" -ne 0 ]
    [[ "$output" == *"no bundled engine found"* ]]
    [ ! -e "${WORK}/uv.args" ]
    run_check_step "${wf}" dist
    [ "$status" -ne 0 ]
    [[ "$output" == *"no bundled engine found"* ]]
    [ ! -e "${WORK}/uv.args" ]
  done
}

@test "every check step fails when the build left two engine trees" {
  for wf in ${WORKFLOWS}; do
    run_check_step "${wf}" dist/a.dist/lilbee_engine_llama_cpp/bin dist/b.dist/lilbee_engine_llama_cpp/bin
    [ "$status" -ne 0 ]
    [[ "$output" == *"a.dist/lilbee_engine_llama_cpp/bin"* ]]
    [[ "$output" == *"b.dist/lilbee_engine_llama_cpp/bin"* ]]
    [ ! -e "${WORK}/uv.args" ]
  done
}

@test "every check step reads the dist tree, not a copy in a build intermediate" {
  for wf in ${WORKFLOWS}; do
    run_check_step "${wf}" "dist/__main__.dist/lilbee_engine_llama_cpp/bin" \
      "dist/__main__.onefile-build/lilbee_engine_llama_cpp/bin" "dist/__main__.build/lilbee_engine_llama_cpp/bin"
    [ "$status" -eq 0 ]
    grep -qF -- '--platform Linux dist/__main__.dist/lilbee_engine_llama_cpp/bin' "${WORK}/uv.args"
  done
}

@test "every check step runs the checker on the one engine tree it finds" {
  for wf in ${WORKFLOWS}; do
    run_check_step "${wf}" "dist/__main__.dist/lilbee_engine_llama_cpp/bin"
    [ "$status" -eq 0 ]
    grep -qF -- '--platform Linux dist/__main__.dist/lilbee_engine_llama_cpp/bin' "${WORK}/uv.args"
    grep -qF -- '--backend vulkan' "${WORK}/uv.args"
  done
}

@test "find_engine_bin.sh prints the single engine tree" {
  mkdir -p "${BATS_TEST_TMPDIR}/dist/x.dist/lilbee_engine_llama_cpp/bin"
  run bash "${REPO_ROOT}/tools/qa/find_engine_bin.sh" "${BATS_TEST_TMPDIR}/dist"
  [ "$status" -eq 0 ]
  [ "$output" = "${BATS_TEST_TMPDIR}/dist/x.dist/lilbee_engine_llama_cpp/bin" ]
}
