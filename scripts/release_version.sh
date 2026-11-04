#!/usr/bin/env bash
# The one place that knows which files carry the lilbee release version and how
# a PEP 440 version becomes the npm (semver) version.
#
#   release_version.sh npm-version 0.6.90b691      -> 0.6.90-beta.691
#   release_version.sh files                       -> the files a release stamps
#   release_version.sh stamp <root> <old> <new>    -> stamp those files under <root>
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
  OLD="$old" NEW="$new" NPM="$npm" perl -pi -e 's/^version = "\Q$ENV{OLD}\E"$/version = "$ENV{NEW}"/' \
    "${root}/pyproject.toml" "${root}/uv.lock"
  # The Claude Code plugin cache keys on the manifest version, so the plugin moves with every release.
  # A source tag from before the plugin existed has no manifest.
  local plugin="${root}/plugins/lilbee/.claude-plugin/plugin.json"
  if [ -f "$plugin" ]; then
    OLD="$old" NEW="$new" perl -pi -e 's/^(\s*"version": ")\Q$ENV{OLD}\E(",?)$/${1}$ENV{NEW}${2}/' "$plugin"
    grep -q "\"version\": \"${new}\"" "$plugin" || fail_stamp plugin.json "$new"
  fi
  local pkg="${root}/packages/lilbee/package.json" lock="${root}/packages/lilbee/package-lock.json"
  if [ -f "$pkg" ]; then
    NPM="$npm" NEW="$new" perl -pi -e 'if (!$d && s/^(  "version": ")[^"]*(",?)$/${1}$ENV{NPM}${2}/) { $d = 1 }
      s/^(    "release": ")[^"]*(",?)$/${1}v$ENV{NEW}${2}/' "$pkg"
    grep -q "^  \"version\": \"${npm}\"," "$pkg" || fail_stamp package.json "$npm"
    grep -q "^    \"release\": \"v${new}\"," "$pkg" || fail_stamp "package.json lilbee.release" "v${new}"
  fi
  if [ -f "$lock" ]; then
    NPM="$npm" perl -0pi -e 's/\A(\{\n  "name": "lilbee",\n  "version": ")[^"]*(")/${1}$ENV{NPM}${2}/;
      s/("packages": \{\n    "": \{\n      "name": "lilbee",\n      "version": ")[^"]*(")/${1}$ENV{NPM}${2}/' "$lock"
    [ "$(head -12 "$lock" | grep -c "\"version\": \"${npm}\",")" = 2 ] || fail_stamp package-lock.json "$npm"
  fi
}

case "${1:-}" in
  npm-version) npm_version "${2:?usage: release_version.sh npm-version <version>}" ;;
  files) printf '%s\n' "${FILES[@]}" ;;
  stamp) stamp "${2:?root}" "${3:?old version}" "${4:?new version}" ;;
  *) echo "usage: release_version.sh npm-version <version> | files | stamp <root> <old> <new>" >&2; exit 2 ;;
esac
