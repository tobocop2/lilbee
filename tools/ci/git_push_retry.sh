#!/usr/bin/env bash
# Push the current branch, rebasing onto whatever landed first when a sibling job
# beat us to the remote.
#
# The package fan-out runs homebrew / scoop / nix-flake / flatpak in parallel, and
# the regular, CUDA and compat workflows run at the same time as each other. Several
# of those jobs commit to the same branch of the same repository, so whoever loses
# the race gets a non-fast-forward rejection and the channel silently goes unpublished
# until someone re-runs the job by hand.
#
# Only safe for text files that different jobs touch independently (a formula, a
# manifest, a flake source). A repository whose content is regenerated wholesale --
# the flatpak OSTree repo -- must serialize its writers instead: rebasing a commit
# that rewrites `summary` and the static deltas would drop whatever the sibling
# published.
#
# Reads:
#   PUSH_REMOTE   remote to push to (default: origin)
#   PUSH_BRANCH   branch to push (default: the checked-out branch, as bare `git push`)
#   PUSH_ATTEMPTS how many times to try (default: 5)

set -euo pipefail

remote="${PUSH_REMOTE:-origin}"
# Not hardcoded to main: the homebrew tap and the flatpak pages repo are separate
# repositories whose default branch is theirs to choose, and the callers reached
# them with a bare `git push`, which follows the checked-out branch.
branch="${PUSH_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
attempts="${PUSH_ATTEMPTS:-5}"

for attempt in $(seq 1 "${attempts}"); do
  if git push "${remote}" "HEAD:${branch}"; then
    exit 0
  fi
  if [ "${attempt}" -eq "${attempts}" ]; then
    break
  fi
  echo "push rejected (attempt ${attempt}/${attempts}); rebasing onto ${remote}/${branch}" >&2
  # A rebase that stops on a conflict would leave the tree mid-rebase and the next
  # push would send the wrong thing. Abort and fail loudly instead: a conflict here
  # means two jobs edited the same file, which this script is not the fix for.
  if ! git pull --rebase "${remote}" "${branch}"; then
    git rebase --abort || true
    echo "rebase onto ${remote}/${branch} conflicted; two jobs are writing the same file" >&2
    exit 1
  fi
  sleep "${attempt}"
done

echo "push to ${remote}/${branch} failed after ${attempts} attempts" >&2
exit 1
