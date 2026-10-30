#!/usr/bin/env bash
# Fail when the locked dependency tree holds a package with a licence that is
# incompatible with lilbee's MIT licence (every GPL and AGPL package).
#
# The input is `uv export` of uv.lock with every extra. The platform and Python
# markers are removed, so one run covers the packages of every OS and Python
# version, not only the installed environment. Pass a requirements file as the
# first argument to check that list instead of the lock.
set -euo pipefail

requirements="${1:-}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

if [ -z "$requirements" ]; then
  requirements="$work/requirements.txt"
  uv export --locked --all-extras --no-hashes --no-annotate --no-dev \
    --no-emit-project --no-emit-package lilbee-engine \
    | sed -e 's/ *;.*//' -e '/^#/d' > "$requirements"
fi

# The checker runs in its own environment, pinned by the lock, so the
# project environment and its native libraries stay out of it.
checker="$work/checker.txt"
uv export --locked --only-group license-check --no-hashes --no-annotate \
  --no-emit-project > "$checker"

run_checker=(uv run --no-project --python 3.12 --with-requirements "$checker")
report="$work/report.json"

# --zero makes licensecheck exit non-zero on an incompatible package. It reads
# a package as compatible when any entry of its licence list is, so the second
# step fails on a GPL or AGPL entry in any list.
status=0
"${run_checker[@]}" licensecheck --requirements-paths "$requirements" \
  --format json --file "$report" --zero || status=$?
"${run_checker[@]}" python -I tools/ci/check_copyleft.py "$report" pyproject.toml || status=$?
exit "$status"
