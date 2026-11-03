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

  # curl answers the registry lookup with REGISTRY_CODE, the npm version document
  # with the next code of NPM_CODES (the last one repeats), and serves the archive.
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
case "${url}" in
  */v0.1/servers/*)
    if [ -n "${CURL_RC:-}" ]; then exit "${CURL_RC}"; fi
    printf '%s' "${REGISTRY_CODE}"
    ;;
  https://registry.npmjs.org/*)
    if [ -n "${NPM_CURL_RC:-}" ]; then exit "${NPM_CURL_RC}"; fi
    n=$(($(cat "${FAKE_NPM_COUNT}" 2>/dev/null || echo 0) + 1))
    echo "${n}" > "${FAKE_NPM_COUNT}"
    code=$(echo "${NPM_CODES}" | cut -d' ' -f"${n}")
    [ -n "${code}" ] || code=$(echo "${NPM_CODES}" | awk '{print $NF}')
    if [ -n "${NPM_MCPNAME}" ]; then
      printf '{"name":"thing-pkg","mcpName":"%s"}' "${NPM_MCPNAME}" > "${out}"
    else
      printf '{"name":"thing-pkg"}' > "${out}"
    fi
    printf '%s' "${code}"
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
  # npm must never be asked: the script reads the version document with curl.
  cat > "${BIN}/npm" <<'STUB'
#!/bin/sh
echo "npm $*" >> "${FAKE_CALLS}"
exit 1
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
  FAKE_CALLS="${CALLS}" FAKE_ARGS="${FIXTURE}/args" FAKE_ARCHIVE="${FIXTURE}/archive.tar.gz" \
  FAKE_NPM_COUNT="${FIXTURE}/npm-count" SLEEP_LOG="${SLEEP_LOG}" MCP_NPM_RETRY_WAIT="${MCP_NPM_RETRY_WAIT:-10}" \
  REGISTRY_CODE="${REGISTRY_CODE:-404}" PATH="${BIN}:${PATH}" \
  NPM_CODES="${NPM_CODES:-200}" NPM_MCPNAME="${NPM_MCPNAME-io.github.example/thing}" \
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
  ! grep -q "^publisher" "${CALLS}" || false
  ! grep -q "^cosign" "${CALLS}" || false
}

@test "a registry error stops the script before any publish" {
  REGISTRY_CODE=503 run publish "${SERVER_JSON}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"answered 503"* ]]
  ! grep -q "^publisher" "${CALLS}" || false
}

@test "a failed signature check stops the script before the publisher runs" {
  COSIGN_RC=1 REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -ne 0 ]
  grep -q "^cosign verify-blob" "${CALLS}"
  ! grep -q "^publisher" "${CALLS}" || false
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
  ! grep -q "^curl" "${CALLS}" || false
}

@test "a transport failure on the registry lookup says so and exits non-zero" {
  CURL_RC=7 run publish "${SERVER_JSON}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"Could not reach the MCP Registry"* ]]
  [[ "$output" == *"curl exit 7"* ]]
  ! grep -q "^publisher" "${CALLS}" || false
}

@test "the registry lookup has a positive time limit" {
  REGISTRY_CODE=200 run publish "${SERVER_JSON}"
  [ "$status" -eq 0 ]
  grep -q -- "^curl-args .*--max-time [1-9][0-9]* .*/v0.1/servers/" "${FIXTURE}/args"
}

@test "npm is asked for the version document of packages[].version, not the packument" {
  printf '{"name": "io.github.example/thing", "version": "1.2.3", "packages": [{"registryType": "npm", "identifier": "@scope/thing", "version": "1.2.4"}]}\n' > "${SERVER_JSON}"
  REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 0 ]
  grep -qx "curl https://registry.npmjs.org/@scope%2Fthing/1.2.4" "${CALLS}"
  ! grep -q "^npm" "${CALLS}" || false
}

@test "the registry lookup is limited to 30 seconds and the npm lookup to 15" {
  REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 0 ]
  grep -q -- "^curl-args .*--max-time 30 .*/v0.1/servers/" "${FIXTURE}/args"
  grep -q -- "^curl-args .*--max-time 15 .*registry.npmjs.org/" "${FIXTURE}/args"
}

@test "a registry lookup that times out says so and exits non-zero" {
  CURL_RC=28 run publish "${SERVER_JSON}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"Could not reach the MCP Registry"* ]]
  [[ "$output" == *"curl exit 28"* ]]
  ! grep -q "^publisher" "${CALLS}" || false
}

@test "a registry code other than 200 and 404 stops before npm is asked" {
  REGISTRY_CODE=429 run publish "${SERVER_JSON}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"answered 429"* ]]
  ! grep -q "registry.npmjs.org" "${CALLS}" || false
}

@test "a listed version is skipped without asking npm and is written to the job summary" {
  GITHUB_STEP_SUMMARY="${FIXTURE}/summary" REGISTRY_CODE=200 run publish "${SERVER_JSON}"
  [ "$status" -eq 0 ]
  ! grep -q "registry.npmjs.org" "${CALLS}" || false
  grep -q "io.github.example/thing 1.2.3 is already in the MCP Registry" "${FIXTURE}/summary"
}

@test "a skip works when the runner gives no job summary file" {
  run env -u GITHUB_STEP_SUMMARY FAKE_CALLS="${CALLS}" FAKE_ARGS="${FIXTURE}/args" FAKE_ARCHIVE="${FIXTURE}/archive.tar.gz" \
    FAKE_NPM_COUNT="${FIXTURE}/npm-count" REGISTRY_CODE=200 PATH="${BIN}:${PATH}" bash "${SCRIPT}" "${SERVER_JSON}"
  [ "$status" -eq 0 ]
  [[ "$output" == *"already in the MCP Registry"* ]]
}

@test "a published package without mcpName is skipped with the reason in log and summary" {
  GITHUB_STEP_SUMMARY="${FIXTURE}/summary" NPM_MCPNAME="" REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 0 ]
  [[ "$output" == *"Skipped the MCP Registry listing"* ]]
  [[ "$output" == *"thing-pkg@1.2.3 has no mcpName"* ]]
  grep -q "thing-pkg@1.2.3 has no mcpName" "${FIXTURE}/summary"
  ! grep -q "^publisher" "${CALLS}" || false
  ! grep -q "^cosign" "${CALLS}" || false
}

@test "a published package with another mcpName goes to the publisher, which owns the refusal" {
  NPM_MCPNAME="io.github.other/thing" REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 0 ]
  grep -qx "publisher publish ${SERVER_JSON}" "${CALLS}"
}

@test "an npm version that appears after two 404s is published after two waits" {
  NPM_CODES="404 404 200" REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 0 ]
  [ "$(grep -c "registry.npmjs.org" "${CALLS}")" -eq 3 ]
  [ "$(cat "${SLEEP_LOG}")" = "$(printf '10\n10')" ]
  grep -qx "publisher publish ${SERVER_JSON}" "${CALLS}"
}

@test "an npm version that stays 404 fails after seven lookups and 60 seconds of waits" {
  NPM_CODES="404" REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"npm has no thing-pkg@1.2.3 after 60 seconds; publish it to npm before listing it"* ]]
  [ "$(grep -c "registry.npmjs.org" "${CALLS}")" -eq 7 ]
  [ "$(wc -l < "${SLEEP_LOG}")" -eq 6 ]
  ! grep -q "^publisher" "${CALLS}" || false
}

@test "the wait between npm lookups is set by MCP_NPM_RETRY_WAIT" {
  NPM_CODES="404 200" REGISTRY_CODE=404 MCP_NPM_RETRY_WAIT=3 run publish "${SERVER_JSON}"
  [ "$status" -eq 0 ]
  [ "$(cat "${SLEEP_LOG}")" = "3" ]
}

@test "an npm answer other than 200 and 404 fails at once without waiting" {
  NPM_CODES="503" REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"npm answered 503"* ]]
  [ ! -s "${SLEEP_LOG}" ]
  ! grep -q "^publisher" "${CALLS}" || false
}

@test "an npm transport failure fails at once without waiting" {
  NPM_CURL_RC=7 REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"Could not reach npm"* ]]
  [[ "$output" == *"curl exit 7"* ]]
  [ ! -s "${SLEEP_LOG}" ]
}

@test "an npm lookup that times out fails at once without waiting" {
  NPM_CURL_RC=28 REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"Could not reach npm"* ]]
  [[ "$output" == *"curl exit 28"* ]]
  [ ! -s "${SLEEP_LOG}" ]
}

@test "a server.json with no npm package is refused" {
  printf '{"name": "io.github.example/thing", "version": "1.2.3"}\n' > "${SERVER_JSON}"
  REGISTRY_CODE=404 run publish "${SERVER_JSON}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"has no npm package"* ]]
}
