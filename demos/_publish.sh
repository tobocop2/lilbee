#!/usr/bin/env bash
# Publish rendered demo GIFs + PNGs to the gh-pages branch (asset store, off main).
# The README and docs reference these via raw.githubusercontent.com URLs.
#
# This uses a dedicated worktree at /tmp/lilbee-gh-pages so the current working tree is
# untouched. Idempotent: re-runs overwrite the prior assets.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
OUT_DIR="$REPO_ROOT/demos/_out"
WORKTREE="${LILBEE_GH_PAGES_WORKTREE:-/tmp/lilbee-gh-pages}"
TARGET_BRANCH="${LILBEE_DEMOS_BRANCH:-gh-pages}"

if [ ! -d "$OUT_DIR" ] || [ -z "$(ls "$OUT_DIR" 2>/dev/null)" ]; then
    echo "error: $OUT_DIR is empty. Run \`make demo\` first." >&2
    exit 1
fi

if [ ! -d "$WORKTREE" ]; then
    echo "==> creating worktree at $WORKTREE for $TARGET_BRANCH"
    git worktree add "$WORKTREE" "$TARGET_BRANCH"
fi

cd "$WORKTREE"
git fetch origin "$TARGET_BRANCH" >/dev/null 2>&1 || true
git checkout "$TARGET_BRANCH"
git pull --ff-only origin "$TARGET_BRANCH" >/dev/null 2>&1 || true

mkdir -p demos
cp -f "$OUT_DIR"/*.gif demos/ 2>/dev/null || true
cp -f "$OUT_DIR"/*.png demos/ 2>/dev/null || true

git add demos/
if git diff --cached --quiet; then
    echo "==> no changes to publish."
    exit 0
fi

git commit -m "demos: refresh rendered GIFs and stills"
echo "==> committed on $TARGET_BRANCH at $WORKTREE."
echo "    push with: (cd $WORKTREE && git push origin $TARGET_BRANCH)"
