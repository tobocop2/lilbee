#!/usr/bin/env bash
# Pre-stage models, data dirs, man pages, and opencode demo dirs for the VHS reel.
# Idempotent: re-running won't re-download or re-index already-staged content.
#
# Env overrides:
#   LILBEE_BIN         which lilbee binary to use (default: first on PATH)
#   LILBEE_DEMO_ROOT   where staged data lives (default: /tmp/lilbee-demo)
#   CV_MANUAL          path to the Crown Vic owner's manual PDF
#                      (default: demos/sample-corpus/cv-manual.pdf in this repo)

set -euo pipefail

DEMOS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DEMOS_DIR
REPO_DIR="$(cd "$DEMOS_DIR/.." && pwd)"
readonly REPO_DIR

readonly LILBEE="${LILBEE_BIN:-lilbee}"
readonly ROOT="${LILBEE_DEMO_ROOT:-/tmp/lilbee-demo}"
readonly MODELS_DIR="$ROOT/models"
readonly CV_MANUAL="${CV_MANUAL:-$DEMOS_DIR/sample-corpus/cv-manual.pdf}"

# Chat model. Pre-staged so chat / catalog / settings / tour tapes start ready.
readonly CHAT_MODEL="Qwen/Qwen3-4B-GGUF"
# Embedding model. Pre-staged for every tape that adds documents.
readonly EMBED_MODEL="nomic-ai/nomic-embed-text-v1.5-GGUF"

# Tapes that record against a pre-indexed copy of the Crown Vic manual.
readonly SEEDED_TAPES=(tui-chat tui-catalog tui-settings tui-tour tui-palette)
# Tapes that start from a clean slate (the recording does its own ingest).
readonly FRESH_TAPES=(tui-setup tui-add tui-crawl cli)
# opencode demo dirs.
readonly OPENCODE_TAPES=(opencode-godot opencode-manual)
# Man pages indexed for the CLI tape.
readonly MAN_PAGES=(find awk grep xargs sed)

log() { printf '==> %s\n' "$*"; }

require_manual() {
    if [[ ! -f "$CV_MANUAL" ]]; then
        printf 'error: %s not found. Set CV_MANUAL=/path/to/manual.pdf.\n' "$CV_MANUAL" >&2
        exit 1
    fi
}

pull_model() {
    local name="$1"
    log "pulling $name (no-op if installed)"
    "$LILBEE" model pull "$name" || true
}

seed_indexed_corpus() {
    # Stage <ROOT>/<tape>/documents/crown-victoria-manual.pdf and rebuild the
    # store, but only if the corpus isn't already there.
    local tape="$1"
    local data="$ROOT/$tape"
    mkdir -p "$data/documents"
    if "$LILBEE" status --data-dir "$data" 2>/dev/null | grep -q crown-victoria-manual; then
        return 0
    fi
    log "indexing manual into $data"
    cp "$CV_MANUAL" "$data/documents/crown-victoria-manual.pdf"
    "$LILBEE" rebuild --data-dir "$data"
}

reset_clean_slate() {
    local tape="$1"
    local data="$ROOT/$tape"
    rm -rf "$data"
    mkdir -p "$data"
}

render_man_page() {
    local page="$1" out_dir="$2"
    local out="$out_dir/$page.txt"
    [[ -f "$out" ]] && return 0
    man "$page" 2>/dev/null | col -bx > "$out" || true
}

page_cache_model() {
    # Read the model file once so macOS keeps it in the page cache. This shaves
    # a few seconds off the next `lilbee` launch's mmap. It does NOT make
    # inference warm: each lilbee process spawns a fresh chat worker with a
    # cold KV cache, so the real warm-up happens inside the tape (`Hide` /
    # warm-up question / `/clear` / `Show`).
    local data="$1"
    "$LILBEE" ask --data-dir "$data" "ping" >/dev/null 2>&1 || true
}

setup_opencode_dir() {
    # Copy the shared agent artifacts (AGENTS.md, opencode.json, the
    # lilbee-worker subagent, and the lilbee-mcp skill) into a demo dir,
    # then `lilbee init` so the MCP server walks up from cwd to a real
    # project-local corpus.
    local tape="$1"
    local dir="$ROOT/$tape"
    mkdir -p "$dir/.opencode/agents" "$dir/.opencode/skills/lilbee-mcp"
    cp "$DEMOS_DIR/AGENTS.md"                                     "$dir/AGENTS.md"
    cp "$DEMOS_DIR/opencode.json"                                 "$dir/opencode.json"
    cp "$DEMOS_DIR/.opencode/agents/lilbee-worker.md"             "$dir/.opencode/agents/lilbee-worker.md"
    cp "$REPO_DIR/docs/agent-skills/lilbee-mcp/SKILL.md"          "$dir/.opencode/skills/lilbee-mcp/SKILL.md"
    rm -rf "$dir/.lilbee"
    ( cd "$dir" && "$LILBEE" init >/dev/null )
}

link_godot_classes() {
    # Copy godot's class reference XMLs into the godot demo dir. The
    # local godot checkout is whatever the dev has at GODOT_SRC (defaults
    # to ~/projects/godot). We copy (not symlink) so the path resolves
    # inside the opencode project root; opencode prompts the user before
    # touching paths outside the project root, which would block the
    # render.
    local dest="$ROOT/opencode-godot/godot-classes"
    [[ -d "$dest" && ! -L "$dest" ]] && return 0
    local godot_src="${GODOT_SRC:-$HOME/projects/godot}"
    if [[ ! -d "$godot_src/doc/classes" ]]; then
        printf 'warn: %s/doc/classes not found; skipping godot demo copy.\n' \
            "$godot_src" >&2
        return 0
    fi
    rm -rf "$dest"
    cp -R "$godot_src/doc/classes" "$dest"
}

main() {
    require_manual

    mkdir -p "$MODELS_DIR"
    export LILBEE_MODELS_DIR="$MODELS_DIR"
    export LILBEE_SKIP_TOML_CONFIG=1

    log "models dir: $MODELS_DIR"
    pull_model "$CHAT_MODEL"
    pull_model "$EMBED_MODEL"

    for tape in "${SEEDED_TAPES[@]}"; do
        seed_indexed_corpus "$tape"
    done

    # Page-cache the chat model so each tape's `lilbee` launch mmaps fast.
    # Inference itself can only be warmed *inside* the tape recording.
    for tape in "${SEEDED_TAPES[@]}"; do
        page_cache_model "$ROOT/$tape"
    done

    for tape in "${FRESH_TAPES[@]}"; do
        reset_clean_slate "$tape"
    done

    local man_dir="$ROOT/cli/man-pages"
    mkdir -p "$man_dir"
    for page in "${MAN_PAGES[@]}"; do
        render_man_page "$page" "$man_dir"
    done

    for tape in "${OPENCODE_TAPES[@]}"; do
        setup_opencode_dir "$tape"
    done
    cp "$CV_MANUAL" "$ROOT/opencode-manual/cv-manual.pdf"
    link_godot_classes

    log "prep done. $ROOT is ready."
}

main "$@"
