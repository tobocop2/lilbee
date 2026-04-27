#!/usr/bin/env bash
# Diagnose the lilbee TUI "illegal hardware instruction" crash on
# pre-AVX2 CPUs (Sandy Bridge / Ivy Bridge / older Xeon). Run this on
# the affected box and paste the entire output back.
#
# What it does:
#   1. Captures CPU + glibc info.
#   2. Sets up a clean venv with lilbee 0.6.66b445 (the AVX-baseline
#      llama_cpp build).
#   3. Runs the TUI launcher under `python -X faulthandler -X dev` so
#      that when SIGILL fires, Python prints the C-extension frame
#      identifying which library crashed.
#   4. Probes pyarrow / numpy / lancedb / tiktoken — the most likely
#      non-llama_cpp SIGILL sources.
#   5. Lists every native .so loaded during a basic lilbee import.
#
# Usage:
#   bash debug-sigill.sh 2>&1 | tee debug-sigill.log
#
# When step 3 hits the TUI, type a message or just let it sit; the
# crash will fire on its own. The script automatically forwards the
# faulthandler trace to the log.

set -uo pipefail

LOG="${LOG:-/tmp/lilbee-debug.log}"
WORKDIR="${WORKDIR:-$HOME/lilbee-debug}"
LILBEE_VERSION="${LILBEE_VERSION:-0.6.66b445}"

banner() {
  printf '\n=== %s ===\n' "$1"
}

banner "01. system / CPU"
uname -a
ldd --version | head -1
echo
echo "CPU model:"
grep -m1 'model name' /proc/cpuinfo
echo "CPU flags (sorted, unique):"
grep '^flags' /proc/cpuinfo | head -1 | tr ' ' '\n' | sort -u | head -50

banner "02. venv setup"
if [ ! -d "$WORKDIR/.venv" ]; then
  mkdir -p "$WORKDIR"
  python -m venv "$WORKDIR/.venv"
fi
PY="$WORKDIR/.venv/bin/python"
PIP="$WORKDIR/.venv/bin/pip"
"$PIP" install --upgrade pip --quiet
"$PIP" install --pre --upgrade --quiet "lilbee==$LILBEE_VERSION"
"$PIP" show lilbee | head -3
echo "python: $($PY -V)"

banner "03. faulthandler-instrumented TUI run (will crash; that's expected)"
echo "Running: $PY -X faulthandler -X dev -m lilbee chat"
echo "Let it run for a few seconds; if the TUI launches, hit CTRL+C twice"
echo "or just wait for the crash. The Python stack at the moment of"
echo "SIGILL will appear inline below."
echo
# Run with stdin closed so the TUI can't block waiting for input;
# faulthandler still prints to stderr on signal.
timeout 30 "$PY" -X faulthandler -X dev -c \
  "from lilbee.launcher import main; main()" </dev/null 2>&1 | tail -200 || true

banner "04. pyarrow probe (lancedb dep — common AVX2 culprit)"
"$PY" -c "
import pyarrow as pa
print('pyarrow:', pa.__version__)
try:
    print('runtime_info:', pa.runtime_info())
except Exception as exc:
    print('runtime_info failed:', exc)
" 2>&1 || true

banner "05. numpy probe"
"$PY" -c "
import numpy as np
print('numpy:', np.__version__)
np.show_config()
" 2>&1 | head -60 || true

banner "06. lancedb probe"
"$PY" -c "
import lancedb
print('lancedb:', lancedb.__version__)
" 2>&1 || true

banner "07. tiktoken probe (Rust-compiled tokenizer)"
"$PY" -c "
import tiktoken
print('tiktoken:', tiktoken.__version__)
enc = tiktoken.get_encoding('cl100k_base')
out = enc.encode('hello world')
print('encoded ok, len:', len(out))
" 2>&1 || true

banner "08. native .so files loaded by `import lilbee.cli` (TUI import chain)"
"$PY" -c "
# Force the same imports the TUI startup does and snapshot the .so map.
import lilbee
from lilbee.cli import app
import os
with open(f'/proc/{os.getpid()}/maps') as f:
    seen = set()
    for line in f:
        parts = line.rstrip().split(maxsplit=5)
        if len(parts) < 6:
            continue
        path = parts[5]
        if not path.endswith('.so') and '.so.' not in path:
            continue
        if path in seen:
            continue
        seen.add(path)
print('=== unique .so files mapped into the TUI process ===')
for p in sorted(seen):
    if 'site-packages' in p or 'lilbee' in p or 'llama' in p:
        print(p)
" 2>&1 || true

banner "09. specific objdump probes for prime suspects"
LIB_DIR=$("$PY" -c "import site; print(site.getsitepackages()[0])")
for pkg_dir in "$LIB_DIR/pyarrow" "$LIB_DIR/numpy" "$LIB_DIR/lancedb" "$LIB_DIR/tiktoken"; do
  [ -d "$pkg_dir" ] || continue
  echo
  echo "--- $pkg_dir ---"
  find "$pkg_dir" -maxdepth 2 -name '*.so' 2>/dev/null | head -5 | while read -r so; do
    echo "  $so"
    objdump -d "$so" 2>/dev/null \
      | grep -oE '\bv(perm|fmadd|broadcast|gather|fnmadd|pmaddub|movntdqa)\w*' \
      | sort | uniq -c | sort -rn | head -3 | sed 's/^/    /'
  done
done

banner "DONE. Paste the entire output (or attach $LOG)."
