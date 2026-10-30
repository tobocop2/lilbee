#!/bin/sh
# Measure every Linux engine on the desktop. Run after fetch_home.sh, under nice.
ROOT=$HOME/lilbee-engines/design-1187
OUT=$ROOT/out
MEASURE="python3 -I $ROOT/measure.py --models $ROOT/models --canaries $ROOT/canaries.json --out $OUT"

server_of() { find "$ROOT/engines/$1" -type f -name llama-server | head -1; }

run_engine() { # engine-label dir-name kind
  bin=$(server_of "$2")
  [ -n "$bin" ] || { echo "NO BINARY $2"; return; }
  chmod +x "$bin"
  lib=$(dirname "$bin")
  LD_LIBRARY_PATH=$lib DYLD_LIBRARY_PATH=$lib "$bin" --list-devices > "$OUT/$1.devices.txt" 2>&1
  $MEASURE --engine "$1" --kind "$3" --server "$bin" --libdir "$lib" --only proof --match 'nomic.*Q4_K_M'
  $MEASURE --engine "$1" --kind "$3" --server "$bin" --libdir "$lib" --corpus "$ROOT/corpus.json" --corpus-limit 400 --match '^(nomic|all-MiniLM|bge)'
  $MEASURE --engine "$1" --kind "$3" --server "$bin" --libdir "$lib" --corpus "$ROOT/corpus.json" --corpus-limit 80 --match '^Qwen3'
}

run_engine cuda cu124 gpu
run_engine vulkan vulkan gpu
run_engine cpu cpu cpu
# The older local CUDA build: a real change of build on the same card.
$MEASURE --engine cuda-build10687 --kind gpu --server "$HOME/gguf45/llama.cpp/build/bin/llama-server" --corpus "$ROOT/corpus.json" --corpus-limit 400 --only base --match '^(nomic|all-MiniLM|bge)'
$MEASURE --engine cuda-build10687 --kind gpu --server "$HOME/gguf45/llama.cpp/build/bin/llama-server" --corpus "$ROOT/corpus.json" --corpus-limit 80 --only base --match '^Qwen3'
echo "MEASURE-HOME-DONE rc=$?"
