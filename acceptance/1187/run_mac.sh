#!/bin/sh
# Measure the macOS engines. Run after fetch_home.sh, under nice.
#   run_mac.sh <scratch dir> <harness dir>
ROOT=$1
HERE=$2
OUT=$ROOT/out
PY=/Users/tobias/projects/lilbee/.venv/bin/python
MEASURE="$PY -I $HERE/measure.py --models $ROOT/models --canaries $HERE/canaries.json --out $OUT"

run_engine() { # engine-label dir-name kind
  bin=$(find "$ROOT/engines/$2" -type f -name llama-server | head -1)
  [ -n "$bin" ] || { echo "NO BINARY $2"; return; }
  chmod +x "$bin"
  lib=$(dirname "$bin")
  LD_LIBRARY_PATH=$lib DYLD_LIBRARY_PATH=$lib "$bin" --list-devices > "$OUT/$1.devices.txt" 2>&1
  $MEASURE --engine "$1" --kind "$3" --server "$bin" --libdir "$lib" --only proof --match 'nomic.*Q4_K_M'
  $MEASURE --engine "$1" --kind "$3" --server "$bin" --libdir "$lib" --corpus "$ROOT/corpus.json" --corpus-limit 400 --match '^(nomic|all-MiniLM|bge)'
  $MEASURE --engine "$1" --kind "$3" --server "$bin" --libdir "$lib" --corpus "$ROOT/corpus.json" --corpus-limit 80 --match '^Qwen3'
}

run_engine metal metal gpu
run_engine maccpu cpu cpu
# The Homebrew build: a real change of build on the same machine.
$MEASURE --engine metal-brew9310 --kind gpu --server /opt/homebrew/bin/llama-server --corpus "$ROOT/corpus.json" --corpus-limit 400 --only base --match '^(nomic|all-MiniLM|bge)'
$MEASURE --engine metal-brew9310 --kind gpu --server /opt/homebrew/bin/llama-server --corpus "$ROOT/corpus.json" --corpus-limit 80 --only base --match '^Qwen3'
echo "MEASURE-MAC-DONE rc=$?"
