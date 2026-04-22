#!/usr/bin/env bash
# Drive the lilbee TUI wiki screen via tmux send-keys and capture panes
# at each step. Exits 0 when every scripted assertion holds, 1 on any
# unexpected-pane diff. Not wired into CI; manual reproduction only.
#
# Requires:
#   - tmux installed (macOS: brew install tmux)
#   - ~/Downloads/cv-manual.pdf (any PDF works; we trim to 10 pages)
#   - a chat model reachable to lilbee (for the regen row; other rows
#     exercise nav / rendering only and run without a model)
#
# Usage:
#   bash tools/qa/wiki_tree_tmux_qa.sh
#
# Artifacts land under /tmp/lilbee-wiki-qa/; re-inspect a failed row's
# pane capture there.

set -uo pipefail

SESSION="lilbee-wiki-qa"
SANDBOX="/tmp/lilbee-wiki-qa"
CAPTURES="$SANDBOX/captures"
SRC_PDF="${HOME}/Downloads/cv-manual.pdf"
TRIMMED_PDF="$SANDBOX/documents/cv-manual-10.pdf"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

fail_count=0
skip_count=0
pass_count=0

log() { printf '%b\n' "$*"; }
pass() { pass_count=$((pass_count + 1)); log "${GREEN}PASS${NC} [$1] $2"; }
skip() { skip_count=$((skip_count + 1)); log "${YELLOW}SKIP${NC} [$1] $2"; }
fail() {
    fail_count=$((fail_count + 1))
    log "${RED}FAIL${NC} [$1] $2"
    [[ -n "${3:-}" ]] && log "     capture: $3"
}

require_tmux() {
    command -v tmux >/dev/null 2>&1 || {
        log "${RED}tmux not installed${NC}. macOS: brew install tmux"
        exit 2
    }
}

setup_sandbox() {
    rm -rf "$SANDBOX"
    mkdir -p "$SANDBOX/documents" "$CAPTURES"

    if [[ ! -f "$SRC_PDF" ]]; then
        log "${RED}source PDF not found${NC}: $SRC_PDF"
        exit 2
    fi

    log "Trimming $SRC_PDF to 10 pages -> $TRIMMED_PDF"
    # pypdf is not a lilbee runtime dep, so we grab it for just this invocation.
    (
        cd "$REPO_ROOT" && uv run --with pypdf python - <<PY
from pathlib import Path
from pypdf import PdfReader, PdfWriter

reader = PdfReader("$SRC_PDF")
writer = PdfWriter()
for page in reader.pages[:10]:
    writer.add_page(page)
Path("$TRIMMED_PDF").parent.mkdir(parents=True, exist_ok=True)
writer.write("$TRIMMED_PDF")
print(f"wrote {len(writer.pages)} pages to $TRIMMED_PDF")
PY
    ) || {
        log "${RED}pdf trim failed${NC}"
        exit 2
    }
}

lilbee_sync() {
    log "Running lilbee sync against the sandbox (wiki enabled)..."
    # LILBEE_DATA sets the data root; documents_dir resolves to $LILBEE_DATA/documents.
    # We drop the trimmed PDF there so the ingest step finds it.
    (
        cd "$REPO_ROOT" \
            && LILBEE_DATA="$SANDBOX" LILBEE_WIKI=true \
               uv run lilbee sync 2>&1 | tail -25
    ) || log "${YELLOW}sync returned non-zero; regen rows may be skipped${NC}"
}

seed_wiki_pages() {
    # Real wiki generation needs a live chat model and takes minutes. The
    # matrix is testing the TUI tree widget, not the generator, so we drop
    # two minimal wiki markdown files directly on disk under the source's
    # nested layout. The Tree sidebar discovers them via list_pages() and
    # renders the Summaries group and its two leaves.
    local pages_dir="$SANDBOX/wiki/summaries/cv-manual-10"
    mkdir -p "$pages_dir"
    cat >"$pages_dir/page-0001.md" <<'MD'
---
title: Page 1 fixture
generated_at: '2026-04-22T00:00:00+00:00'
sources: ["cv-manual-10.pdf"]
faithfulness_score: 0.95
---
# Page 1 fixture

QA fixture content for page 1.
MD
    cat >"$pages_dir/page-0002.md" <<'MD'
---
title: Page 2 fixture
generated_at: '2026-04-22T00:00:00+00:00'
sources: ["cv-manual-10.pdf"]
faithfulness_score: 0.92
---
# Page 2 fixture

QA fixture content for page 2.
MD
    log "Seeded 2 fixture wiki pages at $pages_dir"
}

start_tmux() {
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" -x 180 -y 48
    tmux send-keys -t "$SESSION" \
        "cd '$REPO_ROOT' && LILBEE_DATA='$SANDBOX' LILBEE_WIKI=true uv run lilbee chat" Enter
    sleep 5
}

stop_tmux() {
    tmux kill-session -t "$SESSION" 2>/dev/null || true
}

capture() {
    local name="$1"
    local path="$CAPTURES/$name.txt"
    tmux capture-pane -t "$SESSION" -p >"$path"
    printf '%s' "$path"
}

send() {
    # Send each argument as a separate keystroke with a gap in between so
    # the app's event loop has time to process view transitions, focus
    # changes, etc. before the next key arrives.
    local key
    for key in "$@"; do
        tmux send-keys -t "$SESSION" "$key"
        sleep "${SLEEP:-0.4}"
    done
}

row() {
    local num="$1"; shift
    local label="$1"; shift
    local needle="$1"; shift
    local capture_name="$1"; shift
    # Remaining args are key sequences passed to send(); empty => no-op.
    for keys in "$@"; do
        send "$keys"
    done
    local path
    path=$(capture "$capture_name")
    if grep -qF "$needle" "$path"; then
        pass "$num" "$label"
    else
        fail "$num" "$label (needle: '$needle' not found)" "$path"
    fi
}

run_matrix() {
    # The app-level right_square_bracket binding cycles to the next view with
    # priority=True, so it bubbles out of the chat input. Chat is the default
    # view; five presses cycle to Wiki (Chat -> Catalog -> Status -> Settings
    # -> Tasks -> Wiki). See app.py:Binding("right_square_bracket", ...).
    SLEEP=0.6 send "]" "]" "]" "]" "]"

    # "Summaries" is the tree-group label and stays visible on the wiki
    # sidebar regardless of input focus. "Filter pages..." is the search
    # placeholder and only shows when the search input is empty and not
    # focused, so we avoid it for rows that follow a search interaction.
    #
    # Regenerate (row 8) happens BEFORE the search (row 9) so the tree
    # still has focus when r fires. After row 9's two Escapes the search
    # input is cleared AND focus returns to chat, which row 10 verifies.
    row 1 "Wiki screen opens"       "Summaries"    "01-wiki-open"
    row 2 "Vim down-nav"            "Summaries"    "02-nav-down"    "j" "j" "j"
    row 3 "Vim up-nav"              "Summaries"    "03-nav-up"      "k"
    row 4 "Expand group"            "Summaries"    "04-expand"      "l"
    row 5 "Collapse group"          "Summaries"    "05-collapse"    "h"
    row 6 "Jump bottom then top"    "Summaries"    "06-jump"        "G" "g"
    # Row 6 jumped to the root (collapsed after row 5's h). Re-expand the
    # group and the source branch, then step to the first leaf and press
    # Enter. The fixture leaf's title "Page 1 fixture" should then appear
    # in the right-hand content pane.
    row 7 "Enter renders page"      "Page 1 fixture" "07-enter" "l" "j" "l" "j" "Enter"
    row 8 "Regenerate key bound"    "Summaries"    "08-regen"       "r"
    row 9 "Search filters tree"     "Page 1"       "09-search"      "/" "P" "a" "g" "e" " " "1"
    row 10 "Back to chat via q"     "Ask a"        "10-exit"        "Escape" "Escape"
}

main() {
    require_tmux
    setup_sandbox
    lilbee_sync
    seed_wiki_pages
    start_tmux
    trap stop_tmux EXIT
    run_matrix
    stop_tmux

    log ""
    log "Results: ${GREEN}${pass_count} pass${NC} / ${RED}${fail_count} fail${NC} / ${YELLOW}${skip_count} skip${NC}"
    log "Capture dir: $CAPTURES"
    [[ "$fail_count" -eq 0 ]] || exit 1
}

main "$@"
