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
  chmod +x "${BIN}/curl" "${BIN}/cosign"

  SERVER_JSON="${FIXTURE}/server.json"
  printf '{"name": "io.github.example/thing", "version": "1.2.3"}\n' > "${SERVER_JSON}"
}

publish() {
  FAKE_CALLS="${CALLS}" FAKE_ARCHIVE="${FIXTURE}/archive.tar.gz" \
  REGISTRY_CODE="${REGISTRY_CODE:-404}" PATH="${BIN}:${PATH}" \
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
