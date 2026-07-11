#!/usr/bin/env bash
# Build the Godot 4 class-reference corpus the opencode matrix indexes, once, onto the
# persistent volume. Idempotent: a volume that already has the corpus is left untouched,
# so a pod resume (or a re-launch) skips the embed. Each matrix cell copies this lancedb
# into its workspace, so the reference is embedded exactly once and shared.
set -euo pipefail

CORPUS="${LILBEE_QA_CORPUS:-/workspace/godot_corpus}"
GODOT_DIR="${GODOT_DIR:-/workspace/godot}"

if [ -d "$CORPUS/data/lancedb" ]; then
  echo "[qa_corpus] corpus already built at $CORPUS; skipping"
  exit 0
fi

echo "[qa_corpus] cloning Godot docs (depth 1)"
if [ ! -d "$GODOT_DIR/doc/classes" ]; then
  git clone --depth 1 https://github.com/godotengine/godot "$GODOT_DIR"
fi

echo "[qa_corpus] indexing $GODOT_DIR/doc/classes into $CORPUS (one-time embed)"
LILBEE_DATA="$CORPUS" lilbee add "$GODOT_DIR/doc/classes"
echo "[qa_corpus] done: $CORPUS/data/lancedb"
