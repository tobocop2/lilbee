#!/usr/bin/env bash
# Resolve the version, tag, and ref for a release-candidate run.
#
# Prints version=/tag=/ref= lines on stdout; the caller appends them to
# $GITHUB_OUTPUT. Args: pyproject event ref-name github-ref [input-ref].
set -euo pipefail

pyproject="$1"
event="$2"
ref_name="$3"
github_ref="$4"
input_ref="${5:-}"

version=$(awk -F'"' '/^version *= */ { print $2; exit }' "${pyproject}")
if [ -z "${version}" ]; then
  echo "could not parse version from ${pyproject}" >&2
  exit 1
fi
if [ "${event}" = "push" ]; then
  tag="${ref_name}"
  ref="${github_ref}"
  # Sanity: tag should match pyproject.toml version (allow with-or-without 'v').
  if [ "${tag}" != "v${version}" ] && [ "${tag}" != "${version}" ]; then
    echo "tag ${tag} does not match ${pyproject} version ${version}" >&2
    exit 1
  fi
else
  tag="v${version}"
  ref="${input_ref:-${github_ref}}"
fi
{
  echo "version=${version}"
  echo "tag=${tag}"
  echo "ref=${ref}"
}
