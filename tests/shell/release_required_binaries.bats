#!/usr/bin/env bats
# scripts/assert_required_binaries.sh, the gate that decides whether a dropped
# build cell may strand the release.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/assert_required_binaries.sh"
  DERIVE="${REPO_ROOT}/scripts/release_required_assets.sh"
  WORKFLOW="${REPO_ROOT}/.github/workflows/release.yml"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  mkdir -p "${FIXTURE}/bins" "${FIXTURE}/held"
  # The cells release.yml carries without `soft: true`.
  REQUIRED=(
    lilbee-linux-x86_64
    lilbee-macos-arm64
    lilbee-macos-x86_64
    lilbee-windows-x86_64.exe
  )
  for asset in "${REQUIRED[@]}"; do : > "${FIXTURE}/bins/${asset}"; done
  : > "${FIXTURE}/bins/lilbee-compat-windows-x86_64.exe"
}

assert_binaries() {
  bash "${SCRIPT}" "${FIXTURE}/bins" "${WORKFLOW}"
}

@test "a complete set of executables passes" {
  run assert_binaries
  [ "$status" -eq 0 ]
  [[ "$output" == *"every required executable is present"* ]]
}

@test "a soft cell that dropped out does not block the release" {
  rm "${FIXTURE}/bins/lilbee-compat-windows-x86_64.exe"
  run assert_binaries
  [ "$status" -eq 0 ]
}

@test "every cell without soft is required, and the gate names the one missing" {
  [ "${#REQUIRED[@]}" -eq 4 ]
  local checked=0
  for asset in "${REQUIRED[@]}"; do
    mv "${FIXTURE}/bins/${asset}" "${FIXTURE}/held/${asset}"
    run assert_binaries
    [ "$status" -eq 1 ]
    [[ "$output" == *"never reached"* ]]
    [[ "$output" == *"${asset}"* ]]
    mv "${FIXTURE}/held/${asset}" "${FIXTURE}/bins/${asset}"
    checked=$(( checked + 1 ))
  done
  [ "${checked}" -eq 4 ]
}

# publish-packages.yml cannot publish the default Homebrew formula without the
# Intel macOS binary: Homebrew refuses to load a formula that carries no url for
# the host architecture. A release that drops it is not shippable, so the
# derivation must name it.
@test "the Intel macOS binary is required" {
  # -x: an exact line, so lilbee-compat-macos-x86_64 cannot satisfy the check.
  bash "${DERIVE}" "${WORKFLOW}" | grep -qxF lilbee-macos-x86_64
}

@test "the derivation omits every soft cell" {
  local derived
  derived=$(bash "${DERIVE}" "${WORKFLOW}")
  for asset in lilbee-compat-linux-x86_64 lilbee-compat-macos-x86_64 \
               lilbee-compat-windows-x86_64.exe; do
    run grep -qxF "${asset}" <<< "${derived}"
    [ "$status" -ne 0 ]
  done
}

# The gates that wait on a channel artifact ask this question: a droppable
# binary may never arrive, so nothing that decides whether a tag is promotable
# may wait on what it produces.
@test "the droppable cells are named, and none of them is required" {
  local soft required
  soft=$(bash "${DERIVE}" --soft "${WORKFLOW}")
  required=$(bash "${DERIVE}" "${WORKFLOW}")
  [ -n "${soft}" ]
  local checked=0
  while IFS= read -r asset; do
    run grep -qxF "${asset}" <<< "${required}"
    [ "$status" -ne 0 ]
    checked=$(( checked + 1 ))
  done <<< "${soft}"
  [ "${checked}" -eq 3 ]
}

@test "the compat Linux binary is droppable" {
  # The promotion gate waits for its snap and flatpakref; both are skipped only
  # while this cell stays soft.
  bash "${DERIVE}" --soft "${WORKFLOW}" | grep -qxF lilbee-compat-linux-x86_64
}

@test "a matrix with no droppable cell fails loudly rather than returning nothing" {
  cat > "${FIXTURE}/all-hard.yml" <<'YAML'
jobs:
  build:
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            asset_name: lilbee-linux-x86_64
YAML
  run bash "${DERIVE}" --soft "${FIXTURE}/all-hard.yml"
  [ "$status" -eq 1 ]
  [[ "$output" == *"no cell with soft=true"* ]]
}

@test "the gate and the workflow read the same required set" {
  local derived
  derived=$(bash "${DERIVE}" "${WORKFLOW}" | sort)
  [ "${derived}" = "$(printf '%s\n' "${REQUIRED[@]}" | sort)" ]
}

@test "a cell that lists asset_name before os is still required" {
  # Column-exact text matching dropped such a cell from the required set and
  # left the gate green with the executable missing.
  cat > "${FIXTURE}/reordered.yml" <<'YAML'
jobs:
  build:
    strategy:
        matrix:
            include:
                - asset_name: lilbee-linux-x86_64
                  os: ubuntu-22.04
                - soft: true
                  asset_name: lilbee-compat-windows-x86_64.exe
                  os: windows-latest
YAML
  rm "${FIXTURE}/bins/lilbee-linux-x86_64"
  run bash "${SCRIPT}" "${FIXTURE}/bins" "${FIXTURE}/reordered.yml"
  [ "$status" -eq 1 ]
  [[ "$output" == *"lilbee-linux-x86_64"* ]]
}

@test "a matrix the derivation cannot read fails loudly" {
  printf 'jobs:\n  build:\n    runs-on: ubuntu-latest\n' > "${FIXTURE}/empty.yml"
  run bash "${SCRIPT}" "${FIXTURE}/bins" "${FIXTURE}/empty.yml"
  [ "$status" -eq 1 ]
  [[ "$output" == *"no build cell"* ]]
}
