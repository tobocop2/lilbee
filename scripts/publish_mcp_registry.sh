#!/usr/bin/env bash
# Lists the version in a server.json in the MCP Registry, unless the registry
# already holds it. The registry answers 200 for a listed version and 404 for an
# unlisted one; any other answer, or a failed request, stops the script, so an
# outage never reads as "not listed". The registry refuses an npm package that
# lacks mcpName, so the script lists the version only when the published npm
# package carries the name from server.json. The publisher binary is checked
# against its signature bundle before it runs. Usage: publish_mcp_registry.sh <server.json>
# Env: MCP_REGISTRY_URL overrides the registry address (tests only).
set -euo pipefail

server_json="${1:?usage: publish_mcp_registry.sh <server.json>}"
registry_url="${MCP_REGISTRY_URL:-https://registry.modelcontextprotocol.io}"
publisher_version="v1.8.7"

name=$(jq -r '.name' "${server_json}")
version=$(jq -r '.version' "${server_json}")
[ -n "${name}" ] && [ "${name}" != "null" ] || { echo "${server_json} has no name"; exit 1; }
[ -n "${version}" ] && [ "${version}" != "null" ] || { echo "${server_json} has no version"; exit 1; }

curl_rc=0
code=$(curl --silent --max-time 30 --output /dev/null --write-out '%{http_code}' \
  "${registry_url}/v0.1/servers/${name//\//%2F}/versions/${version}") || curl_rc=$?
if [ "${curl_rc}" -ne 0 ]; then
  echo "Could not reach the MCP Registry to look up ${name} ${version} (curl exit ${curl_rc})."
  exit 1
fi
case "${code}" in
  200)
    echo "${name} ${version} is already in the MCP Registry; nothing to publish."
    exit 0
    ;;
  404) ;;
  *)
    echo "The MCP Registry answered ${code} when asked for ${name} ${version}."
    exit 1
    ;;
esac

package=$(jq -r '[.packages[]? | select(.registryType == "npm")][0].identifier // empty' "${server_json}")
[ -n "${package}" ] || { echo "${server_json} has no npm package"; exit 1; }
npm_rc=0
published_name=$(npm view "${package}@${version}" mcpName) || npm_rc=$?
if [ "${npm_rc}" -ne 0 ]; then
  echo "Could not read mcpName from npm for ${package}@${version} (npm exit ${npm_rc})."
  exit 1
fi
if [ "${published_name}" != "${name}" ]; then
  echo "Skipped the MCP Registry listing: ${package}@${version} has no mcpName ${name} on npm (found: ${published_name:-none}). Bump the version so npm publishes a package that carries it."
  exit 0
fi

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
