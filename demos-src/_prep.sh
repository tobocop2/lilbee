#!/usr/bin/env bash
# Pre-stage models, data dirs, man pages, and opencode demo dirs for the VHS reel.
# Idempotent: re-running won't re-download or re-index already-staged content.
#
# Lives on the `gh-pages` branch alongside the tape sources. Files that live
# on `main` (the agent integration recipes, the lilbee README, the lilbee-mcp
# skill) are read from $LILBEE_REPO_ROOT, which scripts/demo.sh on main sets
# to the main checkout when invoking the gh-pages Makefile.
#
# Env overrides:
#   LILBEE_BIN         which lilbee binary to use (default: first on PATH)
#   LILBEE_DEMO_ROOT   where staged data lives (default: /tmp/lilbee-demo)
#   LILBEE_REPO_ROOT   path to a main checkout for agent recipes + skill
#                      (required; set by scripts/demo.sh on main)
#   CV_MANUAL          path to the Crown Vic owner's manual PDF
#                      (default: demos-src/sample-corpus/cv-manual.pdf here)

set -euo pipefail

DEMOS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DEMOS_DIR
readonly REPO_DIR="${LILBEE_REPO_ROOT:?LILBEE_REPO_ROOT must point at a main checkout}"

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
readonly OPENCODE_TAPES=(opencode-godot opencode-godot-search opencode-manual opencode-self-tune opencode-pdf-self-tune opencode-code-self-tune)
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
    # store, but only if the library isn't already there.
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

seed_ollama_corpus() {
    # tui-ollama-document: index the manual with the OLLAMA embedder into an
    # isolated data dir whose models dir is empty, so the catalog and model
    # bar show only the Ollama-served model (no native models in play).
    # Requires Ollama running with qwen3:0.6b + nomic-embed-text pulled, and a
    # lilbee built with the litellm extra (the remote provider routes there).
    local data="$ROOT/tui-ollama-document"
    local models="$ROOT/tui-ollama-document-models"
    mkdir -p "$data/documents" "$models"
    if "$LILBEE" status --data-dir "$data" 2>/dev/null | grep -q crown-victoria-manual; then
        return 0
    fi
    log "indexing manual via ollama into $data"
    cp "$CV_MANUAL" "$data/documents/crown-victoria-manual.pdf"
    LILBEE_MODELS_DIR="$models" \
        LILBEE_LLM_PROVIDER=remote \
        LILBEE_REMOTE_BASE_URL="http://localhost:11434" \
        LILBEE_CHAT_MODEL="ollama/qwen3:0.6b" \
        LILBEE_EMBEDDING_MODEL="ollama/nomic-embed-text" \
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
    # project-local library.
    local tape="$1"
    local dir="$ROOT/$tape"
    mkdir -p "$dir/.opencode/agents" "$dir/.opencode/skills/lilbee-mcp"
    cp "$REPO_DIR/examples/agent-integration/AGENTS.md"                       "$dir/AGENTS.md"
    cp "$REPO_DIR/examples/agent-integration/opencode.json"                   "$dir/opencode.json"
    cp "$REPO_DIR/examples/agent-integration/.opencode/agents/lilbee-worker.md" "$dir/.opencode/agents/lilbee-worker.md"
    cp "$REPO_DIR/docs/agent-skills/lilbee-mcp/SKILL.md"                       "$dir/.opencode/skills/lilbee-mcp/SKILL.md"
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

seed_self_tune_corpus() {
    # Stage the godot-classes XML reference (the same 810-file pile
    # mcp-godot uses) into opencode-self-tune and index it. The tape
    # asks an A* pathfinding question twice, once at defaults and once
    # after one settings_set call; with defaults the search returns
    # AStar3D only, with tuned retrieval it adds AStarGrid2D and the
    # NavigationPathQueryParameters family, and the visible second
    # answer is measurably richer than the first.
    local dir="$ROOT/opencode-self-tune"
    local src="$ROOT/opencode-godot/godot-classes"
    if [[ ! -d "$src" ]]; then
        printf 'warn: %s missing; run link_godot_classes first.\n' "$src" >&2
        return 0
    fi
    rm -rf "$dir/godot-classes"
    cp -R "$src" "$dir/godot-classes"
    cat >"$dir/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "model": "opencode/qwen3.6-plus-free",
  "permission": {
    "codesearch": "deny",
    "websearch": "deny",
    "webfetch": "deny",
    "read": "allow",
    "write": "allow",
    "edit": "allow",
    "bash": "allow",
    "glob": "allow",
    "grep": "allow",
    "list": "allow",
    "lilbee_*": "allow"
  },
  "mcp": {
    "lilbee": {
      "type": "local",
      "command": ["$LILBEE", "mcp"],
      "timeout": 900000
    }
  }
}
JSON
    if ( cd "$dir" && "$LILBEE" status 2>/dev/null ) | grep -q "Documents:.*[1-9]"; then
        return 0
    fi
    log "indexing godot-classes into $dir (one-time, ~5 min on M1)"
    ( cd "$dir" && "$LILBEE" add ./godot-classes/ )
}

seed_pdf_self_tune_corpus() {
    # Stage the Crown Vic owner's manual PDF into opencode-pdf-self-tune,
    # index it, and pin .lilbee/config.toml to the OLD pre-rebalance
    # defaults (top_k=8, max_distance=0.65, diversity_max_per_source=3).
    # The tape doesn't touch settings up front; the user just asks a
    # natural question and reacts to the thin answer. The agent widens
    # retrieval on its own in turn 2 via lilbee_settings_set.
    local dir="$ROOT/opencode-pdf-self-tune"
    cp -f "$CV_MANUAL" "$dir/cv-manual.pdf"
    mkdir -p "$dir/.lilbee"
    cat >"$dir/.lilbee/config.toml" <<TOML
top_k = "8"
max_distance = "0.65"
diversity_max_per_source = "3"
max_context_sources = "6"
mmr_lambda = "0.2"
TOML
    cat >"$dir/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "model": "opencode/qwen3.6-plus-free",
  "permission": {
    "codesearch": "deny",
    "websearch": "deny",
    "webfetch": "deny",
    "read": "allow",
    "write": "allow",
    "edit": "allow",
    "bash": "allow",
    "glob": "allow",
    "grep": "allow",
    "list": "allow",
    "lilbee_*": "allow"
  },
  "mcp": {
    "lilbee": {
      "type": "local",
      "command": ["$LILBEE", "mcp"],
      "timeout": 900000
    }
  }
}
JSON
    if ( cd "$dir" && "$LILBEE" status 2>/dev/null ) | grep -q "Documents:.*[1-9]"; then
        return 0
    fi
    log "indexing cv-manual.pdf into $dir"
    ( cd "$dir" && "$LILBEE" add ./cv-manual.pdf )
}

seed_code_self_tune_corpus() {
    # Stage lilbee's own src/ into opencode-code-self-tune, index it,
    # and pin .lilbee/config.toml to the OLD pre-rebalance defaults
    # (top_k=8, max_distance=0.65, diversity_max_per_source=3). The
    # tape (mcp-code-self-tune.tape) asks "how does lilbee handle
    # context-window overflow?" twice -- once at OLD defaults, once
    # after the agent self-tunes via lilbee_settings_set. The second
    # answer pulls in chat_worker.py:_window_messages,
    # ContextWindowExceededError, the chat_completions_api 4xx path,
    # and the llama_cpp provider re-raise site.
    #
    # opencode.json points at the local Qwen3-8B Q4_K_M served by
    # lilbee. The model is ~5GB Q4 and takes ~60s to first-load on
    # M1; subsequent turns are ~4-6 min each. See the tape header
    # for the prompt note on why Qwen3-8B needs the explicit
    # lilbee_settings_set cue (a larger model would self-tune on a
    # fully natural "be exhaustive" prompt).
    local dir="$ROOT/opencode-code-self-tune"
    rm -rf "$dir/src"
    cp -R "$REPO_DIR/src" "$dir/src"
    mkdir -p "$dir/.lilbee"
    cat >"$dir/.lilbee/config.toml" <<TOML
top_k = "8"
max_distance = "0.65"
diversity_max_per_source = "3"
max_context_sources = "6"
mmr_lambda = "0.2"
chat_model = "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
TOML
    cat >"$dir/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "model": "lilbee/Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
  "permission": {
    "codesearch": "deny",
    "websearch": "deny",
    "webfetch": "deny",
    "read": "allow",
    "write": "allow",
    "edit": "allow",
    "bash": "allow",
    "glob": "allow",
    "grep": "allow",
    "list": "allow",
    "lilbee_*": "allow"
  },
  "mcp": {
    "lilbee": {
      "type": "local",
      "command": ["$LILBEE", "mcp"],
      "timeout": 900000
    }
  }
}
JSON
    if ( cd "$dir" && "$LILBEE" status 2>/dev/null ) | grep -q "Documents:.*[1-9]"; then
        return 0
    fi
    log "indexing lilbee src/ into $dir (~1 min on M1)"
    ( cd "$dir" && "$LILBEE" add ./src )
}

preindex_godot_corpus() {
    # Pre-index the full 810-XML godot library into the opencode-godot
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

    # tui-ollama-document indexes the manual through Ollama, not a native
    # model. Best-effort: skip if Ollama isn't reachable so prep still
    # completes for the native tapes.
    seed_ollama_corpus || log "skipped tui-ollama-document (ollama not reachable?)"

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
    seed_self_tune_corpus
    seed_pdf_self_tune_corpus
    seed_code_self_tune_corpus

    log "prep done. $ROOT is ready."
}

main "$@"
