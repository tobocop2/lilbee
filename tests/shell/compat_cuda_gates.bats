#!/usr/bin/env bats
# The gates that carry the compat CUDA variant: the GPU cells file, the readers
# of that file, and the workflow inputs and jobs that narrow a run to one variant.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  WORKFLOWS="${REPO_ROOT}/.github/workflows"
  CELLS="${REPO_ROOT}/.github/gpu-cells.yml"
  ASSETS="${REPO_ROOT}/scripts/release_required_assets.sh"
  PLAN="${REPO_ROOT}/scripts/release_plan_cells.sh"
}

# The text of every `run` step of one workflow, with line continuations joined.
runs_of() {  # workflow file name
  local text
  text=$(yq -r '.jobs[].steps[]? | select(.run != null) | .run' "${WORKFLOWS}/$1")
  text="${text//\\$'\n'/ }"
  printf '%s\n' "${text}" | tr -s ' '
}

@test "the cells file requires exactly the cells whose loss blocks a release" {
  run bash "${ASSETS}" "${CELLS}"
  [ "$status" -eq 0 ]
  [ "${output}" = "lilbee-linux-x86_64-cu121
lilbee-linux-x86_64-cu124
lilbee-linux-x86_64-cu125
lilbee-linux-x86_64-rocm
lilbee-compat-linux-x86_64-rocm
lilbee-windows-x86_64-cu124.exe
lilbee-windows-x86_64-cu125.exe" ]
}

@test "the cells file drops only the two compat CUDA cells" {
  run bash "${ASSETS}" --soft "${CELLS}"
  [ "$status" -eq 0 ]
  [ "${output}" = "lilbee-compat-linux-x86_64-cu124
lilbee-compat-windows-x86_64-cu124.exe" ]
}

@test "the planner finds the Windows compat CUDA cell in the cells file" {
  run bash "${PLAN}" lilbee-compat-windows-x86_64-cu124.exe "${CELLS}"
  [ "$status" -eq 0 ]
  [ "$(jq -r 'length' <<< "$output")" -eq 1 ]
  [ "$(jq -r '.[0].os' <<< "$output")" = "windows-2022" ]
  [ "$(jq -r '.[0].soft' <<< "$output")" = "true" ]
}

@test "verify-release reads the required and the droppable cells from the cells file" {
  local text
  text=$(runs_of verify-release.yml)
  [ -n "${text}" ]
  [[ "${text}" == *"release_required_assets.sh .github/gpu-cells.yml"* ]]
  [[ "${text}" == *"release_required_assets.sh --soft .github/gpu-cells.yml"* ]]
}

@test "verify-release waits on the compat CUDA channel files only when their binary is required" {
  local text
  text=$(runs_of verify-release.yml)
  [[ "${text}" == *"lilbee-compat-cuda.flatpakref:lilbee-compat-linux-x86_64-cu124"* ]]
  [[ "${text}" == *"lilbee-compat-linux-x86_64-cu124.snap:lilbee-compat-linux-x86_64-cu124"* ]]
  [[ "${text}" == *'grep -qxF "${pair#*:}" <<< "${soft}" && continue'* ]]
}

@test "the Flatpak publish reads the droppable cells from the cells file" {
  [[ "$(runs_of publish-flatpak.yml)" == *"release_required_assets.sh --soft .github/gpu-cells.yml"* ]]
}

@test "the Flatpak publish step runs only for the app that only_app names" {
  grep -q '^      only_app:$' "${WORKFLOWS}/publish-flatpak.yml"
  run yq -r '.jobs[].steps[]? | select(.uses == "./.github/actions/publish-flatpak") | .if' \
    "${WORKFLOWS}/publish-flatpak.yml"
  [ "$status" -eq 0 ]
  [ "${output}" = "matrix.sha != '' && (inputs.only_app == '' || inputs.only_app == matrix['app-id'])" ]
}

@test "each Docker build step runs only for its own variant" {
  local wf="${WORKFLOWS}/publish-docker.yml" gate
  gate() { yq -r ".jobs[].steps[]? | select(.name == \"$1\") | .if" "${wf}"; }
  [ "$(gate 'Build & push compat CUDA image')" = "inputs.variant == 'compat-cuda'" ]
  run yq -r '.jobs[].steps[]? | select(.uses != null and (.uses | test("build-push-action"))) | .if' "${wf}"
  [ "$(grep -c "inputs.variant == ''" <<< "$output")" -eq 3 ]
  [[ "$output" == *"inputs.variant == 'default'"* ]]
  [[ "$output" == *"inputs.variant == 'cuda'"* ]]
  [[ "$output" == *"inputs.variant == 'rocm'"* ]]
}

@test "the compat CUDA image builds the 12.4 asset on the 12.4 runtime base" {
  run yq -r '.jobs[].steps[]? | select(.name == "Build & push compat CUDA image") | .with["build-args"]' \
    "${WORKFLOWS}/publish-docker.yml"
  [[ "$output" == *"CUDA_BASE=nvidia/cuda:12.4.1-runtime-ubuntu22.04"* ]]
  [[ "$output" == *"CUDA_ASSET=lilbee-compat-linux-x86_64-cu124"* ]]
}

@test "the build workflow passes only_asset from both triggers to the planner" {
  local wf="${WORKFLOWS}/build-gpu-executables.yml"
  [ "$(grep -c '^      only_asset:$' "${wf}")" -eq 2 ]
  run yq -r '.jobs.plan.steps[] | select(.id == "plan") | .env.ONLY_ASSET' "${wf}"
  [ "$output" = '${{ inputs.only_asset }}' ]
  run yq -r '.jobs.plan.steps[] | select(.id == "plan") | .run' "${wf}"
  [[ "$output" == *'release_plan_cells.sh "${ONLY_ASSET}" .github/gpu-cells.yml'* ]]
}

@test "the build matrix is the plan job output" {
  run yq -r '.jobs.build.needs, .jobs.build.strategy.matrix.include' "${WORKFLOWS}/build-gpu-executables.yml"
  [[ "$output" == *"plan"* ]]
  [[ "$output" == *'fromJSON(needs.plan.outputs.cells)'* ]]
}

@test "the plan job checks out the workflow's own ref, never the tag" {
  run yq -r '.jobs.plan.steps[] | select(.uses != null) | .with.ref' "${WORKFLOWS}/build-gpu-executables.yml"
  [ "$output" = '${{ inputs.ref || github.ref }}' ]
}

@test "dispatch-compat-cuda starts the compat CUDA publishers for the tag" {
  local wf="${WORKFLOWS}/release-candidate.yml" text
  text=$(yq -r '.jobs."dispatch-compat-cuda".steps[].run' "${wf}")
  [[ "${text}" == *'gh workflow run publish-compat-cuda-packages.yml --repo "${REPO}" -f tag="${TAG}"'* ]]
  [[ "${text}" == *'gh workflow run publish-docker.yml --repo "${REPO}" -f tag="${TAG}" -f variant=compat-cuda'* ]]
}

@test "dispatch-compat-cuda waits for the CUDA build and runs only for a release tag" {
  local wf="${WORKFLOWS}/release-candidate.yml"
  run yq -r '.jobs."dispatch-compat-cuda".needs[]' "${wf}"
  [[ "$output" == *"build-cuda"* ]]
  run yq -r '.jobs."dispatch-compat-cuda".if' "${wf}"
  [ "$output" = "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')" ]
}
