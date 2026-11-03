#!/usr/bin/env bats
# scripts/publish_mcp_registry.sh against a fake registry, fake cosign and a fake publisher.

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

  # curl answers the registry lookup with REGISTRY_CODE and serves the archive.
  cat > "${BIN}/curl" <<'STUB'
#!/bin/sh
all_args="$*"
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
echo "curl-args ${all_args}" >> "${FAKE_ARGS}"
if [ -n "${CURL_RC:-}" ]; then exit "${CURL_RC}"; fi
case "${url}" in
  */v0.1/servers/*) printf '%s' "${REGISTRY_CODE}" ;;
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
  # npm view answers with the mcpName of the published package, or fails.
  cat > "${BIN}/npm" <<'STUB'
#!/bin/sh
echo "npm $*" >> "${FAKE_CALLS}"
printf '%s\n' "${NPM_MCPNAME}"
[ "${NPM_RC:-0}" -eq 0 ]
STUB
  chmod +x "${BIN}/curl" "${BIN}/cosign" "${BIN}/npm"

  SERVER_JSON="${FIXTURE}/server.json"
  printf '{"name": "io.github.example/thing", "version": "1.2.3", "packages": [{"registryType": "npm", "identifier": "thing-pkg", "version": "1.2.3"}]}\n' > "${SERVER_JSON}"
}

publish() {
  FAKE_CALLS="${CALLS}" FAKE_ARGS="${FIXTURE}/args" FAKE_ARCHIVE="${FIXTURE}/archive.tar.gz" \
  REGISTRY_CODE="${REGISTRY_CODE:-404}" PATH="${BIN}:${PATH}" \
  NPM_MCPNAME="${NPM_MCPNAME-io.github.example/thing}" \
  bash "${SCRIPT}" "$@"
}

@test "an unlisted version is logged in and published from the given file" {
  REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 0 ]
  grep -qx "publisher login github-oidc" "${CALLS}"
  grep -qx "publisher publish ${SERVER_JSON}" "${CALLS}"
}

@test "the registry is asked about the name and version in the file" {
  REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 0 ]
  grep -qx "curl https://registry.modelcontextprotocol.io/v0.1/servers/io.github.example%2Fthing/versions/1.2.3" "${CALLS}"
}

@test "a version the registry already holds is not published again" {
  REGISTRY_CODE=200 run publish "${SERVER_JSON}"
  [ "$status" -eq 0 ]
  [[ "$output" == *"already in the MCP Registry"* ]]
  ! grep -q "^publisher" "${CALLS}"
  ! grep -q "^cosign" "${CALLS}"
}

@test "a registry error stops the script before any publish" {
  REGISTRY_CODE=503 run publish "${SERVER_JSON}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"answered 503"* ]]
  ! grep -q "^publisher" "${CALLS}"
}

@test "a failed signature check stops the script before the publisher runs" {
  COSIGN_RC=1 REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -ne 0 ]
  grep -q "^cosign verify-blob" "${CALLS}"
  ! grep -q "^publisher" "${CALLS}"
}

@test "the signature check names the downloaded archive and its bundle" {
  REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 0 ]
  grep -q "^cosign verify-blob --bundle .*mcp-publisher_linux_amd64.tar.gz.sigstore.json .*mcp-publisher_linux_amd64.tar.gz$" "${CALLS}"
}

@test "a server.json without a version is refused" {
  printf '{"name": "io.github.example/thing"}\n' > "${SERVER_JSON}"
  run publish "${SERVER_JSON}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"has no version"* ]]
  ! grep -q "^curl" "${CALLS}"
}

@test "a transport failure on the registry lookup says so and exits non-zero" {
  CURL_RC=7 run publish "${SERVER_JSON}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"Could not reach the MCP Registry"* ]]
  [[ "$output" == *"curl exit 7"* ]]
  ! grep -q "^publisher" "${CALLS}"
}

@test "the registry lookup has a time limit" {
  REGISTRY_CODE=200 run publish "${SERVER_JSON}"
  [ "$status" -eq 0 ]
  grep -q -- "^curl-args .*--max-time [0-9]* .*/v0.1/servers/" "${FIXTURE}/args"
}

@test "the npm package for this version is asked for its mcpName" {
  REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 0 ]
  grep -qx "npm view thing-pkg@1.2.3 mcpName" "${CALLS}"
}

@test "a published package without mcpName is skipped with the reason" {
  NPM_MCPNAME="" REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 0 ]
  [[ "$output" == *"Skipped the MCP Registry listing"* ]]
  [[ "$output" == *"thing-pkg@1.2.3 has no mcpName"* ]]
  ! grep -q "^publisher" "${CALLS}"
  ! grep -q "^cosign" "${CALLS}"
}

@test "a published package with another mcpName is skipped with the reason" {
  NPM_MCPNAME="io.github.other/thing" REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 0 ]
  [[ "$output" == *"Skipped the MCP Registry listing"* ]]
  [[ "$output" == *"io.github.other/thing"* ]]
  ! grep -q "^publisher" "${CALLS}"
}

@test "a failed npm lookup stops the script instead of skipping" {
  NPM_RC=1 REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"Could not read mcpName from npm"* ]]
  ! grep -q "^publisher" "${CALLS}"
}

@test "a server.json with no npm package is refused" {
  printf '{"name": "io.github.example/thing", "version": "1.2.3"}\n' > "${SERVER_JSON}"
  REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"has no npm package"* ]]
}
