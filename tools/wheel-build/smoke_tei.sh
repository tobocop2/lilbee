#!/usr/bin/env bash
# Start an installed text-embeddings-router and ask it for one embedding.
#
# Usage: smoke_tei.sh <router-binary> <backend>
#
# Every backend must print its version and resolve its libraries (on Linux, with STUB_DIR on
# LD_LIBRARY_PATH when the binary loads libcuda from the GPU driver). Only the cpu backend can serve
# a model on a runner: the CUDA and Metal backends stop after the version and the library check, because
# a runner has no GPU. The cpu run downloads a small public model, starts the router, and waits for
# /health and one /embed answer.
set -euo pipefail

test "$#" -eq 2 || { echo "usage: smoke_tei.sh <router-binary> <backend>" >&2; exit 2; }
router="$1" backend="$2"
model=sentence-transformers/all-MiniLM-L6-v2
port=18080

if [ -n "${STUB_DIR:-}" ]; then
  export LD_LIBRARY_PATH="${STUB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
"${router}" --version
work="$(mktemp -d)"
if [ "$(uname -s)" = Linux ]; then
  ldd "${router}" > "${work}/ldd.txt"
  cat "${work}/ldd.txt"
  if grep -q 'not found' "${work}/ldd.txt"; then echo "the router has an unresolved library"; exit 1; fi
else
  otool -L "${router}"
fi

if [ "${backend}" != cpu ]; then
  echo "UNMEASURED-HW: ${backend} needs a GPU; stopped after the version and the library check"
  exit 0
fi

"${router}" --model-id "${model}" --hostname 127.0.0.1 --port "${port}" > "${work}/router.log" 2>&1 &
pid=$!
trap 'kill "${pid}" 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then break; fi
  kill -0 "${pid}" 2>/dev/null || { cat "${work}/router.log"; echo "the router stopped"; exit 1; }
  sleep 2
done
curl -fsS "http://127.0.0.1:${port}/health" > /dev/null || { cat "${work}/router.log"; echo "no /health answer"; exit 1; }
curl -fsS -X POST "http://127.0.0.1:${port}/embed" -H 'Content-Type: application/json' \
  -d '{"inputs": ["lilbee engine pack check"]}' > "${work}/embed.json"
# The answer is a list with one vector of 384 floats.
python3 - "${work}/embed.json" <<'PY'
import json
import sys

vectors = json.load(open(sys.argv[1], encoding="utf-8"))
assert len(vectors) == 1 and len(vectors[0]) == 384, (len(vectors), len(vectors[0]))
print("embedding length", len(vectors[0]))
PY
