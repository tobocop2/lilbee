#!/usr/bin/env bash
# Read and write the CI mirror: a GitHub release carrying prebuilt build
# artifacts under content-addressed asset names. Used for engine binaries and
# for the Nuitka object cache.
#
# The mirror exists because the Actions cache is the wrong store for these. It
# has a hard 10 GiB per-repo cap with LRU eviction and drops anything unaccessed
# for 7 days, and its saves are scoped to the creating ref, so a release tag can
# neither read what another ref saved nor save anything a later run can read.
# Release assets have none of those limits. Engines are a safe fit because every
# source is pinned in engine-versions.env, so one key always means one set of
# bytes.
#
# curl rather than gh: the manylinux container the Linux release legs build in
# ships no gh CLI.
#
# Usage:
#   ci_mirror.sh asset-name <key>          print the asset name for a key
#   ci_mirror.sh probe      <key>          exit 0 if the mirror carries it
#   ci_mirror.sh fetch      <key> <dir>    extract it into <dir>, or exit 1
#   ci_mirror.sh publish    <key> <dir>    upload <dir> under <key>
#
# Reads:
#   GITHUB_REPOSITORY   owner/repo (set by Actions)
#   GH_TOKEN            token with contents: write; publish only
#   MIRROR_RELEASE_TAG  mirror release tag (default engine-binaries)

set -euo pipefail

readonly DEFAULT_RELEASE_TAG="engine-binaries"
# GitHub rejects release assets over 2 GiB.
readonly MAX_ASSET_BYTES=$((2 * 1024 * 1024 * 1024))
release_tag="${MIRROR_RELEASE_TAG:-${DEFAULT_RELEASE_TAG}}"
repo="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

# Content-addressed like the key it comes from, minus anything a filename or a
# URL would object to. Container images appear in keys and carry slashes and
# colons. Both sides of the mirror derive the name here, because a mismatch
# would not fail, it would silently never hit.
asset_name() {
    printf '%s.tar.gz\n' "$(printf '%s' "${1}" | tr -c 'A-Za-z0-9._-' '_')"
}

download_url() {
    printf 'https://github.com/%s/releases/download/%s/%s\n' "${repo}" "${release_tag}" "$(asset_name "${1}")"
}

api_get() {
    curl -fsSL -H "Authorization: Bearer ${GH_TOKEN}" -H "Accept: application/vnd.github+json" "$@"
}

# The release object's own id is the first in the response; asset ids only
# appear later, nested under "assets".
first_id() {
    grep -o '"id"[[:space:]]*:[[:space:]]*[0-9]*' | head -1 | grep -o '[0-9]*$'
}

resolve_release_id() {
    local id
    id="$(api_get "https://api.github.com/repos/${repo}/releases/tags/${release_tag}" 2>/dev/null | first_id || true)"
    if [ -n "${id}" ]; then
        printf '%s\n' "${id}"
        return 0
    fi
    api_get -X POST "https://api.github.com/repos/${repo}/releases" \
        -d "{\"tag_name\":\"${release_tag}\",\"name\":\"Engine binaries\",\"prerelease\":true,\"body\":\"Content-addressed prebuilt engines consumed by the release builds. Managed by CI; not a user-facing release.\"}" \
        2>/dev/null | first_id || true
}

mirror_carries() {
    local release_id="${1}" asset="${2}"
    api_get "https://api.github.com/repos/${repo}/releases/${release_id}/assets?per_page=100" 2>/dev/null \
        | grep -o '"name"[[:space:]]*:[[:space:]]*"[^"]*"' \
        | grep -qF "\"${asset}\""
}

cmd_asset_name() {
    asset_name "${1:?usage: ci_mirror.sh asset-name <key>}"
}

cmd_probe() {
    local key="${1:?usage: ci_mirror.sh probe <key>}"
    curl -fsI --retry 2 "$(download_url "${key}")" >/dev/null 2>&1
}

cmd_fetch() {
    local key="${1:?usage: ci_mirror.sh fetch <key> <dir>}"
    local dest="${2:?usage: ci_mirror.sh fetch <key> <dir>}"
    local asset tmp
    asset="$(asset_name "${key}")"
    tmp="$(mktemp -d)"

    # -s without -S: a miss is an expected outcome, not something to log as an error.
    if ! curl -fsL --retry 3 --retry-delay 2 -o "${tmp}/${asset}" "$(download_url "${key}")"; then
        echo "Mirror does not carry ${asset}." >&2
        rm -rf "${tmp}"
        return 1
    fi
    mkdir -p "${dest}"
    if ! tar -xzf "${tmp}/${asset}" -C "${dest}"; then
        echo "Mirror asset ${asset} did not extract; leaving ${dest} empty." >&2
        rm -rf "${dest:?}"/*
        return 1
    fi
    rm -rf "${tmp}"
    echo "Restored ${asset}"
}

# Every failure is soft: the mirror is an optimisation with the Actions cache and
# a source build behind it, so a publish problem must not fail a release build.
cmd_publish() {
    local key="${1:?usage: ci_mirror.sh publish <key> <dir>}"
    local src="${2:?usage: ci_mirror.sh publish <key> <dir>}"
    local asset release_id tmp
    asset="$(asset_name "${key}")"

    if [ ! -d "${src}" ] || [ -z "$(ls -A "${src}" 2>/dev/null)" ]; then
        echo "No engine at ${src}; nothing to mirror." >&2
        return 0
    fi

    tmp="$(mktemp -d)"
    if ! tar -czf "${tmp}/${asset}" -C "${src}" .; then
        echo "Could not pack ${src}; skipping the mirror upload." >&2
        return 0
    fi

    local size
    size="$(wc -c < "${tmp}/${asset}" | tr -d ' ')"
    echo "Packed ${asset}: $((size / 1048576)) MB"
    if [ "${size}" -gt "${MAX_ASSET_BYTES}" ]; then
        echo "Packed ${asset} is $((size / 1048576)) MB, over the $((MAX_ASSET_BYTES / 1048576)) MB release-asset limit; skipping the mirror upload." >&2
        return 0
    fi

    release_id="$(resolve_release_id)"
    if [ -z "${release_id}" ]; then
        echo "Could not resolve or create ${release_tag}; skipping the mirror upload." >&2
        return 0
    fi

    if mirror_carries "${release_id}" "${asset}"; then
        echo "Mirror already carries ${asset}."
        return 0
    fi

    if curl -fsSL -X POST \
        -H "Authorization: Bearer ${GH_TOKEN}" \
        -H "Content-Type: application/gzip" \
        --data-binary @"${tmp}/${asset}" \
        "https://uploads.github.com/repos/${repo}/releases/${release_id}/assets?name=${asset}" >/dev/null 2>&1; then
        echo "Mirrored ${asset}."
        rm -rf "${tmp}"
    else
        echo "Mirror upload of ${asset} failed; consumers fall back to rebuilding it." >&2
    fi
}

main() {
    local command="${1:?usage: ci_mirror.sh <asset-name|probe|fetch|publish> ...}"
    shift
    case "${command}" in
        asset-name) cmd_asset_name "$@" ;;
        probe)      cmd_probe "$@" ;;
        fetch)      cmd_fetch "$@" ;;
        publish)    cmd_publish "$@" ;;
        *) echo "unknown command: ${command}" >&2; exit 2 ;;
    esac
}

main "$@"
