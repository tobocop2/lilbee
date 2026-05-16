#!/usr/bin/env bash
# Thin wrapper around the demo-generation pipeline that lives on the
# gh-pages branch (so main stays clean). Worktrees gh-pages, runs the
# corresponding `make` target there, leaves the worktree in place for
# the next call.
#
# Usage:
#   scripts/demo.sh prep      # stages /tmp/lilbee-demo and pre-cached models
#   scripts/demo.sh render    # renders every tape into demos/_out/
#   scripts/demo.sh publish   # commits demos/_out/ + pushes gh-pages

set -euo pipefail

WORKTREE="${LILBEE_GH_PAGES_WORKTREE:-/tmp/lilbee-gh-pages}"
REPO_ROOT="$(git rev-parse --show-toplevel)"

if [ ! -d "$WORKTREE" ]; then
    git -C "$REPO_ROOT" fetch origin gh-pages >/dev/null
    git -C "$REPO_ROOT" worktree add "$WORKTREE" origin/gh-pages
fi

git -C "$WORKTREE" fetch origin gh-pages >/dev/null 2>&1 || true
git -C "$WORKTREE" checkout gh-pages
git -C "$WORKTREE" pull --ff-only origin gh-pages >/dev/null 2>&1 || true

case "${1:-render}" in
    prep)
        make -C "$WORKTREE" demo-prep
        ;;
    render)
        make -C "$WORKTREE" demo
        ;;
    publish)
        make -C "$WORKTREE" demo-publish
        ;;
    *)
        echo "usage: $0 {prep|render|publish}" >&2
        exit 2
        ;;
esac
