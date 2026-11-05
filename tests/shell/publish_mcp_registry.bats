#!/usr/bin/env bats
# scripts/publish_mcp_registry.sh against a fake registry, fake cosign, a fake
# npm and a fake publisher.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/publish_mcp_registry.sh"
  FIXTURE="${BATS_TEST_TMPDIR}/fx"
  BIN="${FIXTURE}/bin"
  CALLS="${FIXTURE}/calls"
  mkdir -p "${BIN}" "${FIXTURE}/payload"
  : > "${CALLS}"

  # The publisher inside the fake archive records what it is asked to do.
  printf '#!/bin/sh\necho "publisher $*" >> "%s"\n' "${CALLS}" > "${FIXTURE}/payload/mcp-publisher"
  chmod +x "${FIXTURE}/payload/mcp-publisher"
  tar czf "${FIXTURE}/archive.tar.gz" -C "${FIXTURE}/payload" mcp-publisher

  # curl answers the registry lookup with REGISTRY_CODE, or fails as a
  # transport with CURL_RC, and serves the publisher archive.
  cat > "${BIN}/curl" <<'STUB'
#!/bin/sh
out=""
url=""
while [ $# -gt 0 ]; do
  case "$1" in
    --output) out="$2"; shift ;;
    http*) url="$1" ;;
  esac
  shift
done
echo "curl ${url}" >> "${FAKE_CALLS}"
case "${url}" in
  */v0.1/servers/*)
    if [ -n "${CURL_RC:-}" ]; then exit "${CURL_RC}"; fi
    printf '%s' "${REGISTRY_CODE}"
    ;;
  *.tar.gz) cp "${FAKE_ARCHIVE}" "${out}" ;;
  *) echo '{}' > "${out}" ;;
esac
STUB
  # cosign passes or fails on demand.
  cat > "${BIN}/cosign" <<'STUB'
#!/bin/sh
echo "cosign $*" >> "${FAKE_CALLS}"
[ "${COSIGN_RC:-0}" -eq 0 ]
STUB
  # npm view is the script's existence probe for the package version. It prints
  # no packument fields, so a package with no mcpName looks like every other.
  cat > "${BIN}/npm" <<'STUB'
#!/bin/sh
echo "npm $*" >> "${FAKE_CALLS}"
exit "${NPM_VIEW_RC:-0}"
STUB
  # sleep records the wait and returns at once.
  cat > "${BIN}/sleep" <<'STUB'
#!/bin/sh
echo "$1" >> "${SLEEP_LOG}"
STUB
  chmod +x "${BIN}/curl" "${BIN}/cosign" "${BIN}/npm" "${BIN}/sleep"
  SLEEP_LOG="${FIXTURE}/sleeps"
  : > "${SLEEP_LOG}"

  SERVER_JSON="${FIXTURE}/server.json"
  printf '{"name": "io.github.example/thing", "version": "1.2.3", "packages": [{"registryType": "npm", "identifier": "thing-pkg", "version": "1.2.3"}]}\n' > "${SERVER_JSON}"
}

publish() {
  FAKE_CALLS="${CALLS}" FAKE_ARCHIVE="${FIXTURE}/archive.tar.gz" \
  SLEEP_LOG="${SLEEP_LOG}" REGISTRY_CODE="${REGISTRY_CODE:-404}" PATH="${BIN}:${PATH}" \
  NPM_VIEW_RC="${NPM_VIEW_RC:-0}" bash "${SCRIPT}" "$@"
}

@test "a version the registry already holds is skipped, with the note in the job summary" {
  GITHUB_STEP_SUMMARY="${FIXTURE}/summary" REGISTRY_CODE=200 run publish "${SERVER_JSON}"
  [ "$status" -eq 0 ]
  [[ "$output" == *"already in the MCP Registry"* ]]
  grep -q "io.github.example/thing 1.2.3 is already in the MCP Registry" "${FIXTURE}/summary"
  ! grep -q "^npm" "${CALLS}" || false
  ! grep -q "^publisher" "${CALLS}" || false
}

@test "an npm package without mcpName still goes to the publisher, which owns the refusal" {
  REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 0 ]
  grep -qx "npm view thing-pkg@1.2.3" "${CALLS}"
  grep -qx "publisher login github-oidc" "${CALLS}"
  grep -qx "publisher publish ${SERVER_JSON}" "${CALLS}"
}

@test "an npm version that never appears stops the script after three attempts" {
  NPM_VIEW_RC=1 REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"npm has no thing-pkg@1.2.3 after 3 attempts"* ]]
  [ "$(grep -cx "npm view thing-pkg@1.2.3" "${CALLS}")" -eq 3 ]
  [ "$(cat "${SLEEP_LOG}")" = "$(printf '10\n10')" ]
  ! grep -q "^publisher" "${CALLS}" || false
}

@test "a registry answer other than 200 and 404 stops the script before npm is asked" {
  REGISTRY_CODE=503 run publish "${SERVER_JSON}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"answered 503"* ]]
  ! grep -q "^npm" "${CALLS}" || false
  ! grep -q "^publisher" "${CALLS}" || false
}

@test "a failed signature check stops the script before the publisher runs" {
  COSIGN_RC=1 REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -ne 0 ]
  grep -q "^cosign verify-blob" "${CALLS}"
  ! grep -q "^publisher" "${CALLS}" || false
}

@test "a server.json with no npm package is refused" {
  printf '{"name": "io.github.example/thing", "version": "1.2.3"}\n' > "${SERVER_JSON}"
  REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"has no npm package"* ]]
  ! grep -q "^npm" "${CALLS}" || false
}
