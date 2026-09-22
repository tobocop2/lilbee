#!/usr/bin/env bash
# Launcher gate for a built executable: --version prints the expected version
# and neither --version nor --help leaks a typer or runtime error onto stderr.
#
# Each probe records its exit code and prints stdout and stderr BEFORE the
# assertion. A probe that fails inside a command substitution under `set -e`
# ends the step at the assignment, and the captured diagnostic is lost.
#
# The launcher is a whole command, so a packaged build reaches its binary
# through its own runner: `flatpak run <app-id>`, a snap path, a plain file.
#
# Usage: assert_launcher_boots.sh <expected-version> <launcher> [launcher-arg...]
set -euo pipefail

usage="assert_launcher_boots.sh: usage: assert_launcher_boots.sh <expected-version> <launcher> [launcher-arg...]"
expected="${1:?${usage}}"
shift
[ "$#" -gt 0 ] || { echo "${usage}" >&2; exit 2; }
launcher=("$@")

# What typer and the frozen multiprocessing dispatch print when the launcher
# mis-detects frozen state and reinvocations reach the CLI parser.
leak_pattern="No such option|Type not yet supported|RuntimeError|ModuleNotFoundError"

stderr_file=$(mktemp)
trap 'rm -f "${stderr_file}"' EXIT

echo "Launcher: ${launcher[*]}"

rc=0
actual=$("${launcher[@]}" --version 2>"${stderr_file}") || rc=$?
echo "--version exit code: ${rc}"
echo "Expected: lilbee ${expected}"
echo "Actual:   ${actual}"
echo "--version stderr:"
cat "${stderr_file}"
if [ "${rc}" -ne 0 ]; then
  echo "FAIL: --version exited ${rc}" >&2
  exit 1
fi
if [ "${actual}" != "lilbee ${expected}" ]; then
  echo "FAIL: --version printed the wrong version" >&2
  exit 1
fi
if grep -qE "${leak_pattern}" "${stderr_file}"; then
  echo "FAIL: --version produced typer/runtime errors on stderr" >&2
  exit 1
fi

rc=0
"${launcher[@]}" --help > /dev/null 2>"${stderr_file}" || rc=$?
echo "--help exit code: ${rc}"
echo "--help stderr:"
cat "${stderr_file}"
if [ "${rc}" -ne 0 ]; then
  echo "FAIL: --help exited ${rc}" >&2
  exit 1
fi
if grep -qE "${leak_pattern}" "${stderr_file}"; then
  echo "FAIL: --help produced typer/runtime errors on stderr" >&2
  exit 1
fi

echo "LAUNCHER OK: --version and --help are clean"
