#!/usr/bin/env bash
# Publishes the npm launcher at a release tag, then lists it in the MCP
# Registry. The tag's own package.json and server.json must be the release the
# tag names, so the check runs before anything is published. A dev build is
# never published. A version npm already holds is not published again, and the
# registry listing still runs, so a re-run after a failed listing lists it.
# Usage: publish_npm.sh <tag>
# Env: NPM_TRUSTED_PUBLISHING must be 'enabled' to publish; any other value
# runs the check and says why nothing published.
set -euo pipefail

tag="${1:?usage: publish_npm.sh <tag>}"
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

case "${tag}" in
  *.dev*)
    echo "${tag} is a dev build; dev builds are not published to npm."
    exit 0
    ;;
esac

cd "${root}"
bash scripts/release_version.sh check "${tag}"

version=$(jq -r '.version' packages/lilbee/package.json)
published=$(npm view lilbee versions --json | jq -r --arg v "${version}" '[.] | flatten | index($v) != null')
echo "lilbee ${version} on npm: ${published}"

if [ "${NPM_TRUSTED_PUBLISHING:-}" != "enabled" ]; then
  echo "Skipped the npm publish and the MCP Registry listing: the repository variable NPM_TRUSTED_PUBLISHING is not 'enabled', so lilbee ${version} did not publish. Finish the one-time npmjs setup in packages/lilbee/docs/publishing.md, then set the variable."
  exit 0
fi

if [ "${published}" = "true" ]; then
  echo "Skipped the npm publish: lilbee ${version} is already on npm."
else
  (cd packages/lilbee && npm publish --access public --tag latest)
fi

# The registry reads mcpName from the published package, so this runs after the publish.
bash scripts/publish_mcp_registry.sh server.json
