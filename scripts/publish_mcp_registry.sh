#!/usr/bin/env bash
# Lists the version in a server.json in the MCP Registry, unless the registry
# already holds it. The registry answers 200 for a listed version and 404 for an
# unlisted one; any other answer stops the script, so an outage never reads as
# "not listed". The publisher binary is checked against its signature bundle
# before it runs. Usage: publish_mcp_registry.sh <server.json>
# Env: MCP_REGISTRY_URL overrides the registry address (tests only).
set -euo pipefail

server_json="${1:?usage: publish_mcp_registry.sh <server.json>}"
registry_url="${MCP_REGISTRY_URL:-https://registry.modelcontextprotocol.io}"
publisher_version="v1.8.7"

name=$(jq -r '.name' "${server_json}")
version=$(jq -r '.version' "${server_json}")
[ -n "${name}" ] && [ "${name}" != "null" ] || { echo "${server_json} has no name"; exit 1; }
[ -n "${version}" ] && [ "${version}" != "null" ] || { echo "${server_json} has no version"; exit 1; }

code=$(curl --silent --output /dev/null --write-out '%{http_code}' \
  "${registry_url}/v0.1/servers/${name//\//%2F}/versions/${version}")
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
