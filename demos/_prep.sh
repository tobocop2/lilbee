#!/usr/bin/env bash
# Pre-stage models, data dirs, man pages, and opencode demo dirs for the VHS reel.
# Idempotent: re-running won't re-download or re-index.

set -euo pipefail

# Use the bundled standalone binary directly. Skips any broken venv install
# (e.g. a pyenv shim with a numpy/thinc mismatch).
LILBEE="${LILBEE_BIN:-/opt/homebrew/bin/lilbee}"

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

# Pull the two demo models once. Re-runs are no-ops if already installed.
echo "==> pulling Qwen/Qwen3-0.6B-GGUF (chat) if missing ..."
"$LILBEE" model pull "Qwen/Qwen3-0.6B-GGUF" || true
echo "==> pulling nomic-ai/nomic-embed-text-v1.5-GGUF (embed) if missing ..."
"$LILBEE" model pull "nomic-ai/nomic-embed-text-v1.5-GGUF" || true

# Pre-index the Crown Vic manual for tapes that need an existing corpus.
# --data-dir goes AFTER the subcommand (lilbee parses flags per-subcommand).
for tape in tui-chat tui-catalog tui-settings tui-tour; do
    DATA="$ROOT/$tape"
    mkdir -p "$DATA/documents"
    unset LILBEE_DATA
    if ! "$LILBEE" status --data-dir "$DATA" 2>/dev/null | grep -q "crown-victoria-manual"; then
        echo "==> indexing manual into $DATA"
        cp "$CV_MANUAL" "$DATA/documents/crown-victoria-manual.pdf"
        "$LILBEE" rebuild --data-dir "$DATA"
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
