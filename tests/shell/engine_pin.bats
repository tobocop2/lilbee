#!/usr/bin/env bats
# The pin of the llama.cpp engine wheel carries its backend on every path that builds the wheel: a mirror hit, a
# cache hit and a fresh build. Two lilbee processes share one engine fleet only when their pins match, so a cu125
# wheel and a Vulkan wheel of the same versions must have different pins.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  BUILD="${REPO_ROOT}/.github/workflows/build-multigpu.yml"
  ACTION="${REPO_ROOT}/.github/actions/bundle-llama-server/action.yml"
  # shellcheck source=/dev/null
  . "${REPO_ROOT}/engine-versions.env"
  BASE_PIN="llama-cpp-${ENGINE_LLAMA_CPP_REF}+swap-${ENGINE_LLAMA_SWAP_VERSION}+gguf-${ENGINE_GGUF_PARSER_REF}"
  # A copy of the files the packaging steps read, so a run never writes into the checkout.
  TREE="${BATS_TEST_TMPDIR}/tree"
  mkdir -p "${TREE}/tools" "${TREE}/packaging"
  cp -R "${REPO_ROOT}/tools/wheel-build" "${TREE}/tools/"
  cp -R "${REPO_ROOT}/packaging/engine-wheel" "${TREE}/packaging/"
  cp "${REPO_ROOT}/engine-versions.env" "${REPO_ROOT}/pyproject.toml" "${TREE}/"
  PINS="${TREE}/packaging/engine-wheel/lilbee_engine_llama_cpp/pins.py"
  # Stubs for the wheel build and the venv install: each records pins.py as the build reads it.
  STUBS="${BATS_TEST_TMPDIR}/stubs"
  SEEN="${BATS_TEST_TMPDIR}/seen-pins.py"
  mkdir -p "${STUBS}"
  printf '#!/bin/sh\ncase "$*" in *"-m build"*) cp lilbee_engine_llama_cpp/pins.py "%s" ;; esac\n' "${SEEN}" \
    > "${STUBS}/python"
  printf '#!/bin/sh\ncp packaging/engine-wheel/lilbee_engine_llama_cpp/pins.py "%s"\n' "${SEEN}" > "${STUBS}/uv"
  chmod +x "${STUBS}/python" "${STUBS}/uv"
}

pin_line() {
  grep '^ENGINE_PIN' "$1"
}

# The run block of the step named $3 of $1/$2, with each ${{ }} expression replaced by a plain word, as the runner
# replaces it before bash reads the block.
step_script() {
  yq -r "$2[] | select(.name == \"$3\") | .run" "$1" | sed 's/\${{[^}]*}}/expr/g'
}

# needs-build as the decide step of $1/$2 sets it, with the mirror and cache results $3 and $4.
needs_build() {
  local script out
  script="$(yq -r "$2[] | select(.id == \"engine\") | .run" "$1")"
  out="${BATS_TEST_TMPDIR}/github-output"
  : > "${out}"
  MIRRORED="$3" CACHED="$4" GITHUB_OUTPUT="${out}" bash -c "${script}"
  sed -n 's/^needs-build=//p' "${out}"
}

# The names of the steps of $1/$2 that run when needs-build is $3, one per line.
running_steps() {
  if [ "$3" = true ]; then
    yq -r "$2[] | .name" "$1"
  else
    yq -r "$2[] | select((.if // \"\") | contains(\"needs-build == 'true'\") | not) | .name" "$1"
  fi
}

# The names of the steps of $1/$2 that write the pin, one per line.
pin_step() {
  yq -r "[$2[] | select((.run // \"\") | contains(\"write_engine_pin.sh\"))] | .[] | .name" "$1"
}

# Run the engine wheel build of build-multigpu.yml on a hit or a fresh build ($1 $2: the mirror and cache results),
# and check that the wheel reads the pin of the backend.
check_multigpu_wheel() {
  local nb step script
  nb="$(needs_build "${BUILD}" '.jobs["build-multigpu"].steps' "$1" "$2")"
  [ "${nb}" = "$3" ]
  step="$(pin_step "${BUILD}" '.jobs["build-multigpu"].steps')"
  [ "${step}" = "Build the lilbee-engine-llama-cpp wheel" ]
  running_steps "${BUILD}" '.jobs["build-multigpu"].steps' "${nb}" | grep -qxF "${step}"
  [ "$(yq -r '.jobs["build-multigpu"].env.BACKEND' "${BUILD}")" = '${{ matrix.backend }}' ]
  [ "$(yq -r '.jobs["build-multigpu"].env.LLAMA_CPP_REF' "${BUILD}")" = '${{ inputs.llama_cpp_ref }}' ]
  script="$(step_script "${BUILD}" '.jobs["build-multigpu"].steps' "${step}")"
  git -C "${TREE}" init -q
  git -C "${TREE}" -c user.name=t -c user.email=t@t commit -q --allow-empty -m tree
  for backend in cu125 vulkan; do
    (cd "${TREE}/packaging/engine-wheel" && PATH="${STUBS}:${PATH}" GITHUB_WORKSPACE="${TREE}" BACKEND="${backend}" \
      LLAMA_CPP_REF="" bash -euo pipefail -c "${script}")
    [ "$(pin_line "${SEEN}")" = "ENGINE_PIN = \"${BASE_PIN}+${backend}\"" ]
  done
}

# The same for the venv install of the bundle-llama-server action.
check_action_install() {
  local nb step script
  nb="$(needs_build "${ACTION}" '.runs.steps' "$1" "$2")"
  [ "${nb}" = "$3" ]
  step="$(pin_step "${ACTION}" '.runs.steps')"
  [ "${step}" = "Install lilbee_engine_llama_cpp into the build venv" ]
  running_steps "${ACTION}" '.runs.steps' "${nb}" | grep -qxF "${step}"
  [ "$(yq -r ".runs.steps[] | select(.name == \"${step}\") | .env.BACKEND" "${ACTION}")" = '${{ inputs.backend }}' ]
  [ "$(yq -r ".runs.steps[] | select(.name == \"${step}\") | .env.LLAMA_CPP_REF" "${ACTION}")" = '${{ inputs.llama-cpp-ref }}' ]
  script="$(step_script "${ACTION}" '.runs.steps' "${step}")"
  for backend in cu125 vulkan; do
    (cd "${TREE}" && PATH="${STUBS}:${PATH}" BACKEND="${backend}" LLAMA_CPP_REF="" bash -euo pipefail -c "${script}")
    [ "$(pin_line "${SEEN}")" = "ENGINE_PIN = \"${BASE_PIN}+${backend}\"" ]
  done
}

@test "the pin script writes the backend into the pin" {
  for backend in cu125 vulkan; do
    run env BACKEND="${backend}" bash "${TREE}/tools/wheel-build/write_engine_pin.sh"
    [ "$status" -eq 0 ]
    [ "$(pin_line "${PINS}")" = "ENGINE_PIN = \"${BASE_PIN}+${backend}\"" ]
  done
}

@test "the pin script names a llama.cpp ref override" {
  run env BACKEND=cu121 LLAMA_CPP_REF=b9665 bash "${TREE}/tools/wheel-build/write_engine_pin.sh"
  [ "$status" -eq 0 ]
  [ "$(pin_line "${PINS}")" = \
    "ENGINE_PIN = \"llama-cpp-b9665+swap-${ENGINE_LLAMA_SWAP_VERSION}+gguf-${ENGINE_GGUF_PARSER_REF}+cu121\"" ]
}

@test "the pin script refuses a run with no backend" {
  before="$(cat "${PINS}")"
  run env -u BACKEND bash "${TREE}/tools/wheel-build/write_engine_pin.sh"
  [ "$status" -eq 1 ]
  [[ "$output" == *"BACKEND is required"* ]]
  [ "$(cat "${PINS}")" = "${before}" ]
}

# The engine binaries are restored from the mirror or the cache in place of this script, so a pin it wrote would be
# skipped on a hit.
@test "the engine build script writes no pin" {
  ! grep -q 'pins.py' "${REPO_ROOT}/tools/wheel-build/build_llama_server.sh"
}

@test "the pin step of each engine wheel build has no condition" {
  run yq -r '.jobs["build-multigpu"].steps[] | select((.run // "") | contains("write_engine_pin.sh")) | .if' "${BUILD}"
  [ "$status" -eq 0 ]
  [ "$output" = "null" ]
  run yq -r '.runs.steps[] | select((.run // "") | contains("write_engine_pin.sh")) | .if' "${ACTION}"
  [ "$status" -eq 0 ]
  [ "$output" = "null" ]
}

@test "a wheel built on a mirror hit carries the backend pin" {
  check_multigpu_wheel true "" false
}

@test "a wheel built on a cache hit carries the backend pin" {
  check_multigpu_wheel false true false
}

@test "a wheel built after a fresh engine build carries the backend pin" {
  check_multigpu_wheel false false true
}

@test "the executable build installs the backend pin on a mirror hit" {
  check_action_install true "" false
}

@test "the executable build installs the backend pin on a cache hit" {
  check_action_install false true false
}

@test "the executable build installs the backend pin after a fresh engine build" {
  check_action_install false false true
}
