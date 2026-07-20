#!/usr/bin/env bash
# Fail if any native library in the build venv is built for a newer macOS than
# the deployment target the standalone binary claims.
#
# Nuitka bundles these dylibs into the onefile, and dyld enforces each one's
# LC_BUILD_VERSION minos independently of the launcher's. A dependency built on
# a macOS 15 runner therefore ships a binary that dies at load on macOS 11-14
# even though the launcher itself is pinned to 11.0 -- the failure lands on the
# user, not on the build, so nothing here catches it without an explicit check.
#
# Usage:
#   MACOSX_DEPLOYMENT_TARGET=11.0 tools/wheel-build/check_macos_floor.sh
# No-op off macOS.
set -euo pipefail

[[ "$(uname -s)" == "Darwin" ]] || exit 0

FLOOR="${MACOSX_DEPLOYMENT_TARGET:?MACOSX_DEPLOYMENT_TARGET is required}"
SITE_PACKAGES="$(uv run --no-sync python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

# Sort -V orders versions, so the greater of {floor, minos} is the last line; a
# minos that is not the floor and sorts above it is newer than the floor.
exceeds_floor() {
    [[ "$1" != "$FLOOR" && "$(printf '%s\n%s\n' "$FLOOR" "$1" | sort -V | tail -1)" == "$1" ]]
}

offenders=""
while IFS= read -r lib; do
    # A Mach-O may carry no LC_BUILD_VERSION (older linkers emit LC_VERSION_MIN_MACOSX
    # instead, and text files caught by the glob carry neither); read both, skip neither-case.
    minos="$(otool -l "$lib" 2>/dev/null | awk '/LC_BUILD_VERSION|LC_VERSION_MIN_MACOSX/{f=1} f&&/minos|version/{print $2; exit}')"
    [[ -n "$minos" ]] || continue
    if exceeds_floor "$minos"; then
        offenders+="  $minos  ${lib#"$SITE_PACKAGES"/}"$'\n'
    fi
done < <(find "$SITE_PACKAGES" \( -name '*.so' -o -name '*.dylib' \) -type f)

if [[ -n "$offenders" ]]; then
    echo "ERROR: these bundled libraries require a newer macOS than the ${FLOOR} floor" >&2
    echo "       this binary claims, so it would fail to launch below their minos:" >&2
    printf '%s' "$offenders" >&2
    exit 1
fi

echo "macOS floor OK: every bundled library loads on ${FLOOR}+"
