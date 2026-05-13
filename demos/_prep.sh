#!/usr/bin/env bash
# Pre-stage models, data dirs, man pages, and opencode demo dirs for the VHS reel.
# Idempotent: re-running won't re-download or re-index.

set -euo pipefail

ROOT="${LILBEE_DEMO_ROOT:-/tmp/lilbee-demo}"
MODELS_DIR="$ROOT/models"
CV_MANUAL="${CV_MANUAL:-$HOME/Downloads/cv-manual.pdf}"

if [ ! -f "$CV_MANUAL" ]; then
    echo "error: $CV_MANUAL not found. Set CV_MANUAL=/path/to/manual.pdf." >&2
    exit 1
fi

mkdir -p "$MODELS_DIR"
export LILBEE_MODELS_DIR="$MODELS_DIR"
export LILBEE_SKIP_TOML_CONFIG=1

echo "==> models dir: $MODELS_DIR"

# Pull the two demo models once into the shared models dir.
# Tapes that show a fresh pull will re-trigger the download in their own (empty) data dir,
# but the cached files in the global HF cache make those quick.
echo "==> pulling qwen3:0.6b (chat) if missing ..."
lilbee model pull qwen3:0.6b || true
echo "==> pulling nomic-embed-text-v1.5 (embed) if missing ..."
lilbee model pull nomic-embed-text-v1.5 || true

# Pre-index the Crown Vic manual for tapes that need an existing corpus.
for tape in tui-chat tui-catalog tui-settings tui-tour; do
    DATA="$ROOT/$tape"
    mkdir -p "$DATA"
    export LILBEE_DATA="$DATA"
    if ! lilbee --data-dir "$DATA" status >/dev/null 2>&1; then
        lilbee --data-dir "$DATA" init || true
    fi
    if ! lilbee --data-dir "$DATA" status 2>/dev/null | grep -q "crown-victoria-manual"; then
        echo "==> indexing manual into $DATA"
        cp "$CV_MANUAL" "$DATA/documents/crown-victoria-manual.pdf" 2>/dev/null || true
        lilbee --data-dir "$DATA" sync || true
    fi
done

# Tapes that record from a clean slate (the setup wizard, the add demo, the CLI tour):
# nuke and recreate their data dirs.
for tape in tui-setup tui-add cli; do
    rm -rf "${ROOT:?}/$tape"
    mkdir -p "$ROOT/$tape"
done

# Man pages for the CLI tape.
MANDIR="$ROOT/cli/man-pages"
mkdir -p "$MANDIR"
for p in find awk grep xargs sed; do
    if [ ! -f "$MANDIR/$p.txt" ]; then
        man "$p" 2>/dev/null | col -bx > "$MANDIR/$p.txt" || true
    fi
done

# opencode demo dirs. Each gets the shared agent artifacts.
DEMOS_DIR="$(cd "$(dirname "$0")" && pwd)"
for tape in opencode-code opencode-manual; do
    DIR="$ROOT/$tape"
    mkdir -p "$DIR/.opencode/agents" "$DIR/.opencode/skills/lilbee-mcp"
    cp "$DEMOS_DIR/AGENTS.md" "$DIR/AGENTS.md"
    cp "$DEMOS_DIR/opencode.json" "$DIR/opencode.json"
    cp "$DEMOS_DIR/.opencode/agents/lilbee-worker.md" "$DIR/.opencode/agents/lilbee-worker.md"
    cp "$DEMOS_DIR/../docs/agent-skills/lilbee-mcp/SKILL.md" "$DIR/.opencode/skills/lilbee-mcp/SKILL.md"
done

# opencode-manual: the PDF.
cp "$CV_MANUAL" "$ROOT/opencode-manual/cv-manual.pdf"

# opencode-code: a shallow clone of lilbee main with src/ at the root.
if [ ! -d "$ROOT/opencode-code/src/lilbee" ]; then
    TMP="$ROOT/opencode-code/_clone"
    rm -rf "$TMP"
    git clone --depth 1 https://github.com/tobocop2/lilbee.git "$TMP" >/dev/null 2>&1
    mv "$TMP/src" "$ROOT/opencode-code/src"
    cp "$TMP/pyproject.toml" "$ROOT/opencode-code/pyproject.toml" 2>/dev/null || true
    rm -rf "$TMP"
fi

echo "==> prep done. $ROOT is ready."
