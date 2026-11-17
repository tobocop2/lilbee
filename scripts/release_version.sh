#!/usr/bin/env bash
# The one place that knows which files carry the lilbee release version and how
# a PEP 440 version becomes the npm (semver) version.
#
#   release_version.sh npm-version 0.6.90b691      -> 0.6.90-beta.691
#   release_version.sh files                       -> the files a release stamps
#   release_version.sh stamp <root> <old> <new>    -> stamp those files under <root>
#   release_version.sh check v0.6.90b691           -> fail unless package.json and server.json are that release
#
# Mapping: X.Y.Z -> X.Y.Z, X.Y.ZbN -> X.Y.Z-beta.N, a trailing .devM adds a
# dev.M prerelease identifier (dev builds are never published to npm).
# Semver compares alphanumeric identifiers as text, so -bN would sort b1000
# before b691; the separate numeric identifier in -beta.N sorts by value.
# Any other form (a, rc, post) has no mapping and is refused.
set -euo pipefail

FILES=(
  pyproject.toml
  uv.lock
  plugins/lilbee/.claude-plugin/plugin.json
  packages/lilbee/package.json
  packages/lilbee/package-lock.json
  server.json
)

npm_version() {
  local version="$1"
  local re='^([0-9]+\.[0-9]+\.[0-9]+)(b([0-9]+))?(\.dev([0-9]+))?$'
  if ! [[ "$version" =~ $re ]]; then
    echo "release_version: '${version}' has no npm version; only X.Y.Z, X.Y.ZbN and a trailing .devN are supported" >&2
    return 1
  fi
  local out="${BASH_REMATCH[1]}" pre=""
  [ -z "${BASH_REMATCH[3]}" ] || pre="beta.$((10#${BASH_REMATCH[3]}))"
  [ -z "${BASH_REMATCH[5]}" ] || pre="${pre:+${pre}.}dev.$((10#${BASH_REMATCH[5]}))"
  echo "${out}${pre:+-${pre}}"
}

fail_stamp() {
  echo "release_version: $1 did not take version $2" >&2
  exit 1
}

stamp() {
  local root="$1" old="$2" new="$3" npm
  npm=$(npm_version "$new")
  OLD="$old" NEW="$new" perl -pi -e 's/^version = "\Q$ENV{OLD}\E"$/version = "$ENV{NEW}"/' \
    "${root}/pyproject.toml" "${root}/uv.lock"
  # The Claude Code plugin cache keys on the manifest version, so the plugin moves with every release.
  # A source tag from before the plugin existed has no manifest.
  local plugin="${root}/plugins/lilbee/.claude-plugin/plugin.json"
  if [ -f "$plugin" ]; then
    OLD="$old" NEW="$new" perl -pi -e 's/^(\s*"version": ")\Q$ENV{OLD}\E(",?)$/${1}$ENV{NEW}${2}/' "$plugin"
    grep -q "\"version\": \"${new}\"" "$plugin" || fail_stamp plugin.json "$new"
  fi
  # npm rewrites package.json and package-lock.json (both version fields) itself.
  if [ -f "${root}/packages/lilbee/package.json" ]; then
    (
      cd "${root}/packages/lilbee"
      npm version "$npm" --no-git-tag-version --allow-same-version > /dev/null
      npm pkg set "lilbee.release=v${new}"
    ) || fail_stamp package.json "$npm"
  fi
  # The MCP Registry listing names the npm version, in the entry and in its npm package.
  if [ -f "${root}/server.json" ]; then
    local listing
    listing=$(jq --arg v "$npm" '.version = $v | .packages[].version = $v' "${root}/server.json") \
      || fail_stamp server.json "$npm"
    printf '%s\n' "$listing" > "${root}/server.json"
  fi
}

# The npm publish job runs this: package.json and server.json must be the release the tag names.
check() {
  local tag="$1" expected version release
  expected=$(npm_version "${tag#v}")
  version=$(node -p 'require("./packages/lilbee/package.json").version')
  release=$(node -p 'require("./packages/lilbee/package.json").lilbee.release')
  if [ "$version" != "$expected" ] || [ "$release" != "$tag" ]; then
    echo "release_version: package.json has version ${version} and lilbee.release ${release}; ${tag} needs version ${expected} and lilbee.release ${tag}" >&2
    return 1
  fi
  local mcp_name
  mcp_name=$(node -p 'require("./packages/lilbee/package.json").mcpName || ""')
  if [ -z "$mcp_name" ] || ! jq -e --arg n "$mcp_name" '.name == $n' server.json > /dev/null; then
    echo "release_version: package.json mcpName '${mcp_name}' must equal the name in server.json" >&2
    return 1
  fi
  if ! jq -e --arg v "$expected" '.version == $v and (.packages | length > 0) and all(.packages[]; .version == $v)' server.json > /dev/null; then
    echo "release_version: server.json does not carry version ${expected} in every version field" >&2
    return 1
  fi
}

case "${1:-}" in
  npm-version) npm_version "${2:?usage: release_version.sh npm-version <version>}" ;;
  files) printf '%s\n' "${FILES[@]}" ;;
  stamp) stamp "${2:?root}" "${3:?old version}" "${4:?new version}" ;;
  check) check "${2:?usage: release_version.sh check <tag>}" ;;
  *) echo "usage: release_version.sh npm-version <version> | files | stamp <root> <old> <new> | check <tag>" >&2; exit 2 ;;
esac
