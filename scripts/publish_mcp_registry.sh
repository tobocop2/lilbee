#!/usr/bin/env bash
# Lists the version in a server.json in the MCP Registry, unless the registry
# already holds it. The registry answers 200 for a listed version and 404 for an
# unlisted one; any other answer, or a failed request, stops the script, so an
# outage never reads as "not listed". The registry owns every refusal about the
# npm package, including a missing or wrong mcpName, so a broken listing is red
# rather than skipped. npm gets three attempts to show a version it has just
# published. The publisher binary is checked against its signature bundle
# before it runs.
# Usage: publish_mcp_registry.sh <server.json>
# Env: MCP_REGISTRY_URL overrides the registry address. GITHUB_STEP_SUMMARY
# gets the skip note.
set -euo pipefail

server_json="${1:?usage: publish_mcp_registry.sh <server.json>}"
registry_url="${MCP_REGISTRY_URL:-https://registry.modelcontextprotocol.io}"
publisher_version="v1.8.7"

name=$(jq -r '.name' "${server_json}")
version=$(jq -r '.version' "${server_json}")
[ -n "${name}" ] && [ "${name}" != "null" ] || { echo "${server_json} has no name"; exit 1; }
[ -n "${version}" ] && [ "${version}" != "null" ] || { echo "${server_json} has no version"; exit 1; }

# Sets lookup_code to the HTTP status and lookup_rc to the curl exit status.
lookup() { # <max seconds> <url> <body file>
  lookup_rc=0
  lookup_code=$(curl --silent --max-time "$1" --output "$3" --write-out '%{http_code}' "$2") || lookup_rc=$?
}

# Prints the line and, when GitHub gives the job a summary file, appends it there.
note() {
  echo "$1"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then echo "$1" >> "${GITHUB_STEP_SUMMARY}"; fi
  return 0
}

lookup 30 "${registry_url}/v0.1/servers/${name//\//%2F}/versions/${version}" /dev/null
if [ "${lookup_rc}" -ne 0 ]; then
  echo "Could not reach the MCP Registry to look up ${name} ${version} (curl exit ${lookup_rc})."
  exit 1
fi
case "${lookup_code}" in
  200)
    note "${name} ${version} is already in the MCP Registry; nothing to publish."
    exit 0
    ;;
  404) ;;
  *)
    echo "The MCP Registry answered ${lookup_code} when asked for ${name} ${version}."
    exit 1
    ;;
esac

# npm must hold the package version before the registry can list it: npm view
# is the existence probe, and three attempts cover npm's replication lag.
package=$(jq -r '[.packages[]? | select(.registryType == "npm")][0].identifier // empty' "${server_json}")
[ -n "${package}" ] || { echo "${server_json} has no npm package"; exit 1; }
package_version=$(jq -r --arg v "${version}" '[.packages[]? | select(.registryType == "npm")][0].version // $v' "${server_json}")
npm_attempts=3
npm_retry_wait=10
npm_seen=""
for attempt in $(seq 1 "${npm_attempts}"); do
  if npm view "${package}@${package_version}" > /dev/null 2>&1; then
    npm_seen=1
    break
  fi
  if [ "${attempt}" -lt "${npm_attempts}" ]; then sleep "${npm_retry_wait}"; fi
done
[ -n "${npm_seen}" ] || {
  echo "npm has no ${package}@${package_version} after ${npm_attempts} attempts; publish it to npm before listing it."
  exit 1
}

workdir=$(mktemp -d)
archive="mcp-publisher_linux_amd64.tar.gz"
base_url="https://github.com/modelcontextprotocol/registry/releases/download/${publisher_version}"
curl --fail --location --silent --show-error --retry 3 --output "${workdir}/${archive}" "${base_url}/${archive}"
curl --fail --location --silent --show-error --retry 3 --output "${workdir}/${archive}.sigstore.json" "${base_url}/${archive}.sigstore.json"
cosign verify-blob \
  --bundle "${workdir}/${archive}.sigstore.json" \
  --certificate-identity "https://github.com/modelcontextprotocol/registry/.github/workflows/release.yml@refs/tags/${publisher_version}" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  "${workdir}/${archive}"
tar xzf "${workdir}/${archive}" -C "${workdir}" mcp-publisher

"${workdir}/mcp-publisher" login github-oidc
"${workdir}/mcp-publisher" publish "${server_json}"
