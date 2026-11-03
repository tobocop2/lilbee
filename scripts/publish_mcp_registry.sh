#!/usr/bin/env bash
# Lists the version in a server.json in the MCP Registry, unless the registry
# already holds it. The registry answers 200 for a listed version and 404 for an
# unlisted one; any other answer, or a failed request, stops the script, so an
# outage never reads as "not listed". The registry refuses an npm package that
# lacks mcpName or names another server, so the script skips a package with no
# mcpName and leaves every other refusal to the registry. It waits up to a
# minute for npm to show a version it has just published. The publisher binary
# is checked against its signature bundle before it runs.
# Usage: publish_mcp_registry.sh <server.json>
# Env: MCP_REGISTRY_URL overrides the registry address, MCP_NPM_RETRY_WAIT the
# seconds between npm lookups (tests only). GITHUB_STEP_SUMMARY gets each skip.
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

# Ask npm for the document the registry asks for: the package at packages[].version.
package=$(jq -r '[.packages[]? | select(.registryType == "npm")][0].identifier // empty' "${server_json}")
[ -n "${package}" ] || { echo "${server_json} has no npm package"; exit 1; }
package_version=$(jq -r --arg v "${version}" '[.packages[]? | select(.registryType == "npm")][0].version // $v' "${server_json}")
workdir=$(mktemp -d)
npm_url="https://registry.npmjs.org/${package//\//%2F}/${package_version}"
npm_waits=6
npm_wait="${MCP_NPM_RETRY_WAIT:-10}"
for attempt in $(seq 0 "${npm_waits}"); do
  lookup 15 "${npm_url}" "${workdir}/npm.json"
  if [ "${lookup_rc}" -ne 0 ]; then
    echo "Could not reach npm to look up ${package}@${package_version} (curl exit ${lookup_rc})."
    exit 1
  fi
  [ "${lookup_code}" = "404" ] || break
  if [ "${attempt}" -lt "${npm_waits}" ]; then sleep "${npm_wait}"; fi
done
case "${lookup_code}" in
  200) ;;
  404)
    echo "npm has no ${package}@${package_version} after $((npm_waits * npm_wait)) seconds; publish it to npm before listing it."
    exit 1
    ;;
  *)
    echo "npm answered ${lookup_code} when asked for ${package}@${package_version}."
    exit 1
    ;;
esac
if [ -z "$(jq -r '.mcpName // empty' "${workdir}/npm.json")" ]; then
  note "Skipped the MCP Registry listing: ${package}@${package_version} has no mcpName on npm. Bump the version so npm publishes a package that carries it."
  exit 0
fi

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
