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
readonly FRESH_TAPES=(tui-setup tui-add tui-crawl)
# opencode demo dirs.
readonly OPENCODE_TAPES=(opencode-godot opencode-godot-search opencode-manual)
# Subset of godot-classes used by the small live-indexing demo. Six
# files = ~110 KB; indexes in seconds. Keeps mcp-godot-search.tape from
# stalling on indexing dead air.
readonly GODOT_SEARCH_FILES=(
    AStarGrid2D.xml
    AStar2D.xml
    AStar3D.xml
    Vector2.xml
    Vector2i.xml
    Rect2i.xml
)
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

seed_setup_models() {
    # Pre-pull the chat + embedding models into the tui-setup tape's
    # isolated LILBEE_MODELS_DIR so the wizard reliably shows "Your
    # existing models are ready". Real cold-pull renders are too
    # sensitive to bandwidth -- a slow connection leaves the wizard at
    # <100% when the chat fires and the demo captures a "Model not
    # found in registry" error.
    local setup_models="$ROOT/setup-models"
    mkdir -p "$setup_models"
    log "pre-caching chat + embedder in setup-models"
    LILBEE_MODELS_DIR="$setup_models" "$LILBEE" model pull "$CHAT_MODEL" || true
    LILBEE_MODELS_DIR="$setup_models" "$LILBEE" model pull "$EMBED_MODEL" || true
    # Drop any prior SmolLM cache so the tape demonstrates a real pull
    # of SmolLM2 135M live in the Task Center.
    rm -rf "$setup_models/models--bartowski--SmolLM2-135M-Instruct-GGUF"
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
    # The agent writes .gd / .tscn / project.godot files during the godot
    # demos. Wipe them so each re-render starts from "no pre-existing
    # level generator" and the agent has to create from scratch.
    find "$dir" -maxdepth 1 -type f \( -name "*.gd" -o -name "*.tscn" -o -name "project.godot" \) -delete
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

stage_godot_search_subset() {
    # Copy a 6-file pathfinding subset of godot-classes into the
    # opencode-godot-search demo dir; the live mcp-godot-search tape
    # indexes these on camera in ~3 seconds.
    local src="$ROOT/opencode-godot/godot-classes"
    local dest="$ROOT/opencode-godot-search/godot-pathfinding"
    [[ -d "$src" ]] || return 0
    mkdir -p "$dest"
    local f
    for f in "${GODOT_SEARCH_FILES[@]}"; do
        cp -f "$src/$f" "$dest/$f"
    done
}

preindex_godot_corpus() {
    # Pre-index the full 810-XML godot corpus into the opencode-godot
    # demo dir so mcp-godot.tape opens straight into the cited-answer
    # phase. Skips if already indexed (mtime gate makes lilbee add a
    # near-no-op on rerun anyway, but the explicit check shaves the
    # filesystem walk + manifest read).
    local dir="$ROOT/opencode-godot"
    [[ -d "$dir/godot-classes" ]] || return 0
    if "$LILBEE" status --data-dir "$dir" 2>/dev/null | grep -q "Documents:.*[1-9]"; then
        return 0
    fi
    log "pre-indexing godot-classes (one-time, ~5 min on M1)"
    ( cd "$dir" && "$LILBEE" add ./godot-classes/ )
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

    # tui-setup's first chat is `/add ./README.md` against lilbee's own README,
    # so stage it in the demo's data dir after the clean-slate reset.
    cp "$REPO_DIR/README.md" "$ROOT/tui-setup/README.md"
    seed_setup_models

    for tape in "${OPENCODE_TAPES[@]}"; do
        setup_opencode_dir "$tape"
    done
    cp "$CV_MANUAL" "$ROOT/opencode-manual/cv-manual.pdf"
    link_godot_classes
    stage_godot_search_subset
    preindex_godot_corpus

    log "prep done. $ROOT is ready."
}

main "$@"
