#!/usr/bin/env bash
# Land a rewritten flake sources file on main through an auto-merging pull request.
#
# Main is protected, so the pin goes to flake-pins/<package>-<version>, one branch
# per package and version. The branch push uses the checkout's deploy key, whose
# events start CI; the pull request is opened with GH_TOKEN, whose events do not,
# so a new pull request gets one more deploy-key push for its CI to run on.
# Each package owns one sources file, so pin pull requests of different packages
# never conflict. A rerun rebuilds its branch on the current main and force-pushes
# it, and a newer version closes the older open pin pull request of its package.
#
# Run from a checkout of main with the rewritten file in the working tree.
#
# Reads:
#   TARGET_FILE     the sources file (sources.json or sources-<variant>.json)
#   VERSION         the release version, without a leading v
#   COMMIT_MESSAGE  the pin commit's message and the pull request's title
#   DRY_RUN         true: print the branch, commit and pull request; push nothing

set -euo pipefail

target="${TARGET_FILE:?TARGET_FILE is required}"
version="${VERSION:?VERSION is required}"
message="${COMMIT_MESSAGE:?COMMIT_MESSAGE is required}"
dry_run="${DRY_RUN:-false}"
base=main
retry="$(cd "$(dirname "$0")" && pwd)/ci_retry.sh"

case "${target}" in
  sources.json) package=lilbee ;;
  sources-*.json) package="lilbee-${target#sources-}"; package="${package%.json}" ;;
  *)
    echo "flake_pin_pr.sh: ${target} is not a flake sources file" >&2
    exit 2
    ;;
esac
branch="flake-pins/${package}-${version}"
echo "branch: ${branch}"

git add -- "${target}"
if git diff --cached --quiet; then
  echo "no source changes; nothing to publish"
  exit 0
fi
git checkout -q -B "${branch}"
git commit -q -m "${message}"
echo "commit: $(git log -1 --format='%h %s')"

pr=$(gh pr list --head "${branch}" --state open --json number --jq '.[0].number // empty')
# True when version $1 is older than version $2. A version is major.minor.patch
# with an optional bN beta suffix, and a beta is older than its release. One
# that does not parse is never older, so its pull request stays open.
older_than() {
  jq -n -e --arg a "$1" --arg b "$2" '
    def key: capture("^(?<m>[0-9]+)\\.(?<n>[0-9]+)\\.(?<p>[0-9]+)(b(?<b>[0-9]+))?$")
      | [(.m | tonumber), (.n | tonumber), (.p | tonumber), (if .b then (.b | tonumber) else infinite end)];
    ($a | key) < ($b | key)' > /dev/null 2>&1
}

# A version starts with a digit, so lilbee-<v> never matches lilbee-cuda-<v>.
# Only older versions are superseded: republishing an old tag leaves a newer pin alone.
open_pins=$(gh pr list --state open --limit 200 --json number,headRefName \
  --jq ".[] | select(.headRefName | test(\"^flake-pins/${package}-[0-9]\")) | \"\(.number) \(.headRefName)\"")
superseded=""
while read -r n name; do
  [ -n "${n}" ] || continue
  if older_than "${name#"flake-pins/${package}-"}" "${version}"; then
    superseded="${superseded} ${n}"
  fi
done <<< "${open_pins}"

if [ "${dry_run}" = "true" ]; then
  git show --stat --format= HEAD
  if [ -n "${pr}" ]; then
    echo "would update pull request #${pr} and enable auto-merge"
  else
    echo "would open a pull request from ${branch} to ${base} and enable auto-merge"
  fi
  for n in ${superseded}; do echo "would close pull request #${n}: superseded"; done
  echo "dry run: nothing pushed"
  exit 0
fi

bash "${retry}" 3 git push --force origin "HEAD:refs/heads/${branch}"

if [ -z "${pr}" ]; then
  gh pr create --base "${base}" --head "${branch}" --title "${message}" \
    --body "Pins the Nix flake's \`${target}\` to the ${version} release assets. It merges itself once the required check passes."
  git commit -q --allow-empty -m "Run CI on the ${package} ${version} pin"
  bash "${retry}" 3 git push origin "HEAD:refs/heads/${branch}"
fi

gh pr merge "${branch}" --auto --squash

for n in ${superseded}; do
  gh pr close "${n}" --delete-branch --comment "Superseded by ${branch}."
done
