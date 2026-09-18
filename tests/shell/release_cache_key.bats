#!/usr/bin/env bats
# scripts/strip_own_version.sh, the normalization the Nuitka object-cache key hashes.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/strip_own_version.sh"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  mkdir -p "${FIXTURE}"
  lock 0.6.90b442 3.14.3 > "${FIXTURE}/uv.lock"
}

lock() {  # own-version aiohttp-version
  cat <<LOCK
version = 1
requires-python = ">=3.11"

[[package]]
name = "aiohttp"
version = "$2"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "lilbee"
version = "$1"
source = { editable = "." }
dependencies = [
    { name = "aiohttp" },
]

[[package]]
name = "yarl"
version = "1.22.0"
source = { registry = "https://pypi.org/simple" }
LOCK
}

strip() {  # source destination
  bash "${SCRIPT}" "$1" "$2"
}

@test "the lilbee version line is the only line dropped" {
  run strip "${FIXTURE}/uv.lock" "${FIXTURE}/deps.lock"
  [ "$status" -eq 0 ]
  [ "$(wc -l < "${FIXTURE}/uv.lock")" -eq 20 ]
  [ "$(wc -l < "${FIXTURE}/deps.lock")" -eq 19 ]
  ! grep -q '0.6.90b442' "${FIXTURE}/deps.lock"
  grep -q '^version = "3.14.3"$' "${FIXTURE}/deps.lock"
  grep -q '^version = "1.22.0"$' "${FIXTURE}/deps.lock"
}

@test "the key input is unchanged by a version-only release bump" {
  strip "${FIXTURE}/uv.lock" "${FIXTURE}/before.lock"
  lock 0.6.90b443 3.14.3 > "${FIXTURE}/uv.lock"
  strip "${FIXTURE}/uv.lock" "${FIXTURE}/after.lock"
  run cmp "${FIXTURE}/before.lock" "${FIXTURE}/after.lock"
  [ "$status" -eq 0 ]
}

@test "the key input moves when a dependency version moves" {
  strip "${FIXTURE}/uv.lock" "${FIXTURE}/before.lock"
  lock 0.6.90b442 3.14.4 > "${FIXTURE}/uv.lock"
  strip "${FIXTURE}/uv.lock" "${FIXTURE}/after.lock"
  run cmp -s "${FIXTURE}/before.lock" "${FIXTURE}/after.lock"
  [ "$status" -ne 0 ]
}

@test "the unnormalized lock moves on a version-only bump, which is what rotated the key" {
  cp "${FIXTURE}/uv.lock" "${FIXTURE}/before.lock"
  lock 0.6.90b443 3.14.3 > "${FIXTURE}/uv.lock"
  run cmp -s "${FIXTURE}/before.lock" "${FIXTURE}/uv.lock"
  [ "$status" -ne 0 ]
}

@test "the real lock gives up exactly one line" {
  run strip "${REPO_ROOT}/uv.lock" "${FIXTURE}/real.lock"
  [ "$status" -eq 0 ]
  own=$(awk -F'"' '/^version *= */ { print $2; exit }' "${REPO_ROOT}/pyproject.toml")
  grep -q "^version = \"${own}\"\$" "${REPO_ROOT}/uv.lock"
  ! grep -q "^version = \"${own}\"\$" "${FIXTURE}/real.lock"
}

@test "a lock checked out with CRLF line endings still normalizes" {
  sed 's/$/\r/' "${FIXTURE}/uv.lock" > "${FIXTURE}/crlf.lock"
  run strip "${FIXTURE}/crlf.lock" "${FIXTURE}/deps.lock"
  [ "$status" -eq 0 ]
  ! grep -q '0.6.90b442' "${FIXTURE}/deps.lock"
  grep -q '3.14.3' "${FIXTURE}/deps.lock"
}

@test "a lock with no lilbee entry fails rather than keying on everything" {
  grep -v '^name = "lilbee"$' "${FIXTURE}/uv.lock" > "${FIXTURE}/nameless.lock"
  run strip "${FIXTURE}/nameless.lock" "${FIXTURE}/deps.lock"
  [ "$status" -eq 1 ]
  [[ "$output" == *"expected exactly 1"* ]]
}
