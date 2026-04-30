#!/usr/bin/env bash
# Automated QA matrix for the lilbee frozen binary.
# Exercises CLI / HTTP / MCP / TUI surfaces with clear pass/fail criteria.
# Usage: bash tools/qa-binary.sh <path-to-binary>

set -uo pipefail

BIN="${1:?usage: $0 <path-to-binary>}"
[ -x "$BIN" ] || { echo "FAIL: $BIN is not executable"; exit 2; }

WORKDIR=$(mktemp -d -t lilbee-qa-XXXXXX)
trap 'cleanup' EXIT INT TERM

cleanup() {
    [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null || true
    [ -n "${TUI_TMUX:-}" ] && tmux kill-session -t "$TUI_TMUX" 2>/dev/null || true
    rm -rf "$WORKDIR"
}

export LILBEE_DATA="$WORKDIR/data"
export LILBEE_DOCUMENTS_DIR="$WORKDIR/docs"
mkdir -p "$LILBEE_DATA" "$LILBEE_DOCUMENTS_DIR"

PASS=0
FAIL=0
CHECK_LOG="$WORKDIR/checks.log"

# pass <name>
pass() { echo "[PASS] $1"; PASS=$((PASS+1)); }
# fail <name> <reason>
fail() { echo "[FAIL] $1: $2"; FAIL=$((FAIL+1)); }
# section <name>
section() { echo; echo "=== $1 ==="; }

# Read expected version from the binary itself for self-consistency.
EXPECTED_VERSION=$("$BIN" --version 2>/dev/null | awk '{print $2}')
if [ -z "$EXPECTED_VERSION" ]; then
    fail "version-baseline" "binary --version produced no output"
    echo
    echo "SUMMARY: 0 passed, 1 failed (baseline aborted)"
    exit 1
fi

# ============================================================
section "CLI"
# ============================================================

# 1. --version
OUT=$("$BIN" --version 2>&1)
if echo "$OUT" | grep -qE "^lilbee [0-9]"; then
    pass "cli.version-shape"
else
    fail "cli.version-shape" "got: $OUT"
fi

# 2. --version warm cache (second invocation; threshold is generous because
# whole-binary import-graph init still happens, but ought to be much faster
# than the ~15s PyInstaller cold pattern that bb-nsl3 was about).
WARM_MS=$(python3 -c "
import subprocess, time
t = time.perf_counter()
subprocess.run(['$BIN', '--version'], capture_output=True)
print(round((time.perf_counter() - t) * 1000))
")
if [ "$WARM_MS" -le 4000 ]; then
    pass "cli.version-warm-${WARM_MS}ms"
else
    fail "cli.version-warm" "warm run took ${WARM_MS}ms, expected <=4000ms"
fi

# 3. --help
"$BIN" --help >"$WORKDIR/help.out" 2>&1
RC=$?
if [ "$RC" = "0" ]; then
    pass "cli.help-exit-0"
else
    fail "cli.help-exit-0" "exit=$RC"
fi

# 4. --help lists key subcommands
if grep -q "search" "$WORKDIR/help.out" && grep -q "serve" "$WORKDIR/help.out" && grep -q "self-check" "$WORKDIR/help.out"; then
    pass "cli.help-shows-subcommands"
else
    fail "cli.help-shows-subcommands" "missing search/serve/self-check"
fi

# 5. self-check-extras --json
"$BIN" --json self-check-extras >"$WORKDIR/extras.json" 2>&1
RC=$?
if [ "$RC" = "0" ]; then
    pass "cli.self-check-extras-exit-0"
else
    fail "cli.self-check-extras-exit-0" "exit=$RC; output: $(cat "$WORKDIR/extras.json")"
fi

# 6. self-check-extras reports all 4 extras true
if python3 -c "
import json, sys
data = json.loads(open('$WORKDIR/extras.json').read().strip().splitlines()[-1])
required = ['litellm', 'crawl4ai', 'spacy', 'graspologic_native']
missing = [r for r in required if not data.get(r)]
sys.exit(1 if missing else 0)
" 2>/dev/null; then
    pass "cli.self-check-extras-all-true"
else
    fail "cli.self-check-extras-all-true" "missing: $(cat "$WORKDIR/extras.json")"
fi

# 7. --json status returns valid JSON
"$BIN" --json status >"$WORKDIR/status.json" 2>&1
if python3 -c "import json; d = json.loads(open('$WORKDIR/status.json').read().strip().splitlines()[-1]); assert d.get('command') == 'status'" 2>/dev/null; then
    pass "cli.status-json-shape"
else
    fail "cli.status-json-shape" "$(cat "$WORKDIR/status.json")"
fi

# 8. add a test markdown file
TESTDOC="$WORKDIR/testdoc.md"
TOKEN="quokka-rendezvous-$(date +%s)"
cat >"$TESTDOC" <<EOF
# Test Doc

This document contains the test token: $TOKEN
It exists to validate end-to-end ingest+search through the frozen binary.
EOF

"$BIN" --json add "$TESTDOC" >"$WORKDIR/add.json" 2>&1
RC=$?
if [ "$RC" = "0" ]; then
    pass "cli.add-exit-0"
else
    fail "cli.add-exit-0" "exit=$RC; $(tail -5 "$WORKDIR/add.json")"
fi

# 9. status reports the new doc
"$BIN" --json status >"$WORKDIR/status2.json" 2>&1
if python3 -c "
import json
d = json.loads(open('$WORKDIR/status2.json').read().strip().splitlines()[-1])
sources = d.get('sources', [])
assert any('testdoc' in s.get('filename', '') for s in sources), f'testdoc not in sources: {sources}'
" 2>/dev/null; then
    pass "cli.status-shows-added-doc"
else
    fail "cli.status-shows-added-doc" "$(cat "$WORKDIR/status2.json")"
fi

# 10. search finds the unique token
"$BIN" --json search "$TOKEN" >"$WORKDIR/search.json" 2>&1
if python3 -c "
import json
d = json.loads(open('$WORKDIR/search.json').read().strip().splitlines()[-1])
results = d.get('results', [])
assert any('$TOKEN' in r.get('chunk', '') for r in results), f'token not found in results: {len(results)} results'
" 2>/dev/null; then
    pass "cli.search-finds-token"
else
    fail "cli.search-finds-token" "$(cat "$WORKDIR/search.json")"
fi

# ============================================================
section "HTTP"
# ============================================================

PORT=$((50000 + RANDOM % 10000))
"$BIN" serve --port "$PORT" >"$WORKDIR/server.log" 2>&1 &
SERVER_PID=$!

# Wait for server with a 30-second budget
for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        fail "http.server-start" "server died; tail: $(tail -10 "$WORKDIR/server.log")"
        break
    fi
    sleep 1
done

if kill -0 "$SERVER_PID" 2>/dev/null; then
    pass "http.server-start"

    # 1. /api/health
    HEALTH=$(curl -sf "http://127.0.0.1:$PORT/api/health" 2>/dev/null)
    if echo "$HEALTH" | python3 -c "import json, sys; d = json.loads(sys.stdin.read()); assert d.get('version') == '$EXPECTED_VERSION', f'version mismatch: {d}'" 2>/dev/null; then
        pass "http.health-version-match"
    else
        fail "http.health-version-match" "got: $HEALTH"
    fi

    # 2. /api/status
    STATUS=$(curl -sf "http://127.0.0.1:$PORT/api/status" 2>/dev/null)
    if echo "$STATUS" | python3 -c "import json, sys; json.loads(sys.stdin.read())" 2>/dev/null; then
        pass "http.status-valid-json"
    else
        fail "http.status-valid-json" "got: $STATUS"
    fi

    # 3. /api/search?q=<token> (response shape: list of {source, excerpts: [{content}]})
    SEARCH=$(curl -sf "http://127.0.0.1:$PORT/api/search?q=$TOKEN" 2>/dev/null)
    if echo "$SEARCH" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
results = d if isinstance(d, list) else d.get('results', [])
def found(r, t):
    if t in r.get('chunk', '') or t in r.get('content', ''):
        return True
    return any(t in ex.get('content', '') for ex in r.get('excerpts', []))
assert any(found(r, '$TOKEN') for r in results), f'token not in {len(results)} results'
" 2>/dev/null; then
        pass "http.search-finds-token"
    else
        fail "http.search-finds-token" "got: $SEARCH"
    fi

    # 4. SIGTERM shutdown is clean
    kill -TERM "$SERVER_PID" 2>/dev/null
    SHUTDOWN_OK=0
    for _ in $(seq 1 10); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            SHUTDOWN_OK=1
            break
        fi
        sleep 1
    done
    if [ "$SHUTDOWN_OK" = "1" ]; then
        pass "http.sigterm-clean-shutdown"
    else
        fail "http.sigterm-clean-shutdown" "still alive after 10s SIGTERM, sending SIGKILL"
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
fi

# ============================================================
section "MCP"
# ============================================================

# Send a JSON-RPC initialize via stdio. MCP servers respond with capabilities.
MCP_REQ='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"qa-harness","version":"0.0.1"}}}'

MCP_OUT=$(echo "$MCP_REQ" | timeout 15 "$BIN" mcp 2>"$WORKDIR/mcp.stderr" | head -1)
if echo "$MCP_OUT" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
assert d.get('jsonrpc') == '2.0'
assert d.get('id') == 1
assert 'result' in d, f'no result: {d}'
" 2>/dev/null; then
    pass "mcp.initialize-handshake"
else
    fail "mcp.initialize-handshake" "stdout: $MCP_OUT; stderr tail: $(tail -3 "$WORKDIR/mcp.stderr")"
fi

# ============================================================
section "TUI"
# ============================================================

if ! command -v tmux >/dev/null 2>&1; then
    echo "[SKIP] tui.* (tmux not installed)"
else
    TUI_TMUX="lilbee-qa-$$"
    tmux new-session -d -s "$TUI_TMUX" -x 200 -y 50 "$BIN" 2>/dev/null
    sleep 5
    tmux capture-pane -t "$TUI_TMUX" -p >"$WORKDIR/tui.out" 2>&1
    if grep -qE "Chat|Catalog|Status|lilbee" "$WORKDIR/tui.out"; then
        pass "tui.chrome-rendered"
    else
        fail "tui.chrome-rendered" "missing UI chrome; pane: $(head -20 "$WORKDIR/tui.out")"
    fi

    # Quit cleanly (Ctrl-Q is the standard textual quit binding)
    tmux send-keys -t "$TUI_TMUX" "C-q" 2>/dev/null
    sleep 2
    if ! tmux has-session -t "$TUI_TMUX" 2>/dev/null; then
        pass "tui.quit-clean"
    else
        fail "tui.quit-clean" "session still alive after Ctrl-Q"
        tmux kill-session -t "$TUI_TMUX" 2>/dev/null
    fi
    TUI_TMUX=""
fi

# ============================================================
section "CACHE"
# ============================================================

# Nuitka --product-version is the int-tuple form (e.g. 0.6.66.456 from
# pyproject 0.6.66b456). The cache dir uses that form, not the PEP 440 string.
CACHE_DIR_BASE="$HOME/.cache/lilbee"
NUITKA_VERSION=$(python3 -c "
import re, sys
parts = re.findall(r'\d+', '$EXPECTED_VERSION')
print('.'.join(parts[:4]))
")
if [ -d "$CACHE_DIR_BASE/$NUITKA_VERSION" ]; then
    pass "cache.dir-exists"
else
    fail "cache.dir-exists" "expected $CACHE_DIR_BASE/$NUITKA_VERSION (binary version $EXPECTED_VERSION); have: $(ls "$CACHE_DIR_BASE" 2>/dev/null | tr '\n' ' ')"
fi

# ============================================================
section "SUMMARY"
# ============================================================

TOTAL=$((PASS+FAIL))
echo
echo "RESULTS: $PASS passed, $FAIL failed (out of $TOTAL)"
echo "WORKDIR: $WORKDIR (kept on failure for inspection)"

if [ "$FAIL" = "0" ]; then
    echo "QA MATRIX: GREEN"
    rm -rf "$WORKDIR"
    trap - EXIT INT TERM
    exit 0
fi

# Keep workdir for postmortem on failure
trap - EXIT INT TERM
exit 1
