#!/usr/bin/env bash
# Publishes the npm launcher at a release tag, then lists it in the MCP
# Registry. The tag's own package.json and server.json must be the release the
# tag names, so the check runs before anything is published. A version npm
# already holds is not published again, and the registry listing still runs, so
# a re-run after a failed listing lists it.
# The script exits 0 only when npm and the registry serve the version. A dev
# tag and an unset NPM_TRUSTED_PUBLISHING fail it: the workflow does not
# request npm for a dev tag, so a request that cannot publish is an error.
# Usage: publish_npm.sh <tag>
# Env: NPM_TRUSTED_PUBLISHING must be 'enabled'.
set -euo pipefail

tag="${1:?usage: publish_npm.sh <tag>}"
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

case "${tag}" in
  *.dev*)
    echo "${tag} is a dev build. Dev builds are not published to npm, so do not request the npm channel for one." >&2
    exit 1
    ;;
esac

cd "${root}"
bash scripts/release_version.sh check "${tag}"

version=$(jq -r '.version' packages/lilbee/package.json)
published=$(npm view lilbee versions --json | jq -r --arg v "${version}" '[.] | flatten | index($v) != null')
echo "lilbee ${version} on npm: ${published}"

if [ "${NPM_TRUSTED_PUBLISHING:-}" != "enabled" ]; then
  echo "The repository variable NPM_TRUSTED_PUBLISHING is not 'enabled', so lilbee ${version} did not publish to npm or the MCP Registry. Finish the one-time npmjs setup in packages/lilbee/docs/publishing.md, then set the variable." >&2
  exit 1
fi

if [ "${published}" = "true" ]; then
  echo "Skipped the npm publish: lilbee ${version} is already on npm."
else
  (cd packages/lilbee && npm publish --access public --tag latest)
fi

# The registry reads mcpName from the published package, so this runs after the publish.
bash scripts/publish_mcp_registry.sh server.json
