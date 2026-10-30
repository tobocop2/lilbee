#!/bin/sh
# Fetch the shipped engine wheels and the measured GGUFs into one host's scratch tree.
#   fetch_home.sh <scratch dir> <wheel suffix ...>
# Wheel URLs and model URLs are fixed, so that a later lane measures the same files.
set -u
ROOT=$1
shift
TAG=v0.6.90b691
BASE=https://github.com/tobocop2/lilbee/releases/download/$TAG
HF=https://huggingface.co
mkdir -p "$ROOT/wheels" "$ROOT/models" "$ROOT/engines"

get() { # url dest
  [ -s "$2" ] && { echo "HAVE $2"; return 0; }
  curl -fsSL --retry 5 --retry-delay 5 -C - -o "$2.part" "$1" && mv "$2.part" "$2" && echo "GOT $2" || echo "FAILED $1"
}

for suffix in "$@"; do
  w=lilbee_engine-0.6.90b691-1.$suffix.whl
  get "$BASE/$w" "$ROOT/wheels/$w"
  d=$ROOT/engines/${suffix%%-*}
  mkdir -p "$d"
  python3 -I -m zipfile -e "$ROOT/wheels/$w" "$d" && echo "UNPACKED $d"
done

while read -r repo file; do
  [ -n "$repo" ] || continue
  get "$HF/$repo/resolve/main/$file" "$ROOT/models/$file"
done <<EOF
nomic-ai/nomic-embed-text-v1.5-GGUF nomic-embed-text-v1.5.Q4_K_M.gguf
nomic-ai/nomic-embed-text-v1.5-GGUF nomic-embed-text-v1.5.Q8_0.gguf
nomic-ai/nomic-embed-text-v1.5-GGUF nomic-embed-text-v1.5.Q2_K.gguf
nomic-ai/nomic-embed-text-v1.5-GGUF nomic-embed-text-v1.5.f16.gguf
second-state/All-MiniLM-L6-v2-Embedding-GGUF all-MiniLM-L6-v2-Q4_K_M.gguf
second-state/All-MiniLM-L6-v2-Embedding-GGUF all-MiniLM-L6-v2-Q8_0.gguf
second-state/All-MiniLM-L6-v2-Embedding-GGUF all-MiniLM-L6-v2-Q2_K.gguf
CompendiumLabs/bge-base-en-v1.5-gguf bge-base-en-v1.5-q4_k_m.gguf
CompendiumLabs/bge-base-en-v1.5-gguf bge-base-en-v1.5-q8_0.gguf
Qwen/Qwen3-Embedding-0.6B-GGUF Qwen3-Embedding-0.6B-Q8_0.gguf
Qwen/Qwen3-Embedding-0.6B-GGUF Qwen3-Embedding-0.6B-f16.gguf
EOF
echo "FETCH-DONE rc=0"
