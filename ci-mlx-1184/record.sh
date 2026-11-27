#!/usr/bin/env bash
# Harness only: start the chat, embed and rerank servers of the pack that pack_mlx.sh unpacked into check/, and
# record the strict output check, as tests/conformance/fixtures/engines/mlx-metal.json was recorded on macOS 27.
set -euo pipefail

pkg="${GITHUB_WORKSPACE}/check/lilbee_engine_mlx"
py="${pkg}/python/bin/python3"
logs="${RUNNER_TEMP}/logs"
out="${RUNNER_TEMP}/out"
mkdir -p "${logs}" "${out}"
chat_id=mlx-community/Qwen3-0.6B-4bit
embed_id=mlx-community/Qwen3-Embedding-0.6B-8bit
rerank_id=mlx-community/Qwen3-Reranker-0.6B-4bit

snap() {  # repo
  "${py}" -I -c 'import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1]))' "$1"
}
chat="$(snap "${chat_id}")"
embed="$(snap "${embed_id}")"
rerank="$(snap "${rerank_id}")"

# All three servers write one log, so each listening process holds the file that the recorder reads.
log="${logs}/engine.log"
: > "${log}"
"${pkg}/bin/mlx-lm-server" --model "${chat}" --host 127.0.0.1 --port 8081 --max-tokens 32768 >> "${log}" 2>&1 &
"${pkg}/bin/mlx-embed-server" --model "${embed}" --host 127.0.0.1 --port 8082 --max-length 32768 --task embed \
  >> "${log}" 2>&1 &
"${pkg}/bin/mlx-embed-server" --model "${rerank}" --host 127.0.0.1 --port 8083 --max-length 32768 --task rerank \
  >> "${log}" 2>&1 &
for port in 8081 8082 8083; do
  n=0
  until nc -z 127.0.0.1 "${port}"; do
    n=$((n + 1))
    [ "${n}" -lt 300 ] || { echo "port ${port} did not open"; tail -40 "${log}"; exit 1; }
    sleep 2
  done
done
echo "SERVERS UP"

uv sync -q --locked --extra providers --python 3.12
uv run --no-sync python -m tests.conformance.record_presets \
  --server mlx-metal --version 0.32.2 --engine mlx --backend metal --strict --window 32768 \
  --url http://127.0.0.1:8081 --embed-url http://127.0.0.1:8082 --rerank-url http://127.0.0.1:8083 \
  --roles chat,embed,rerank --embed-dim 1024 \
  --models "chat=${chat_id},embed=${embed_id},rerank=${rerank_id}" \
  --engine-log "${log}" --out "${out}/mlx-metal-macos15.json" 2>&1 | tee "${logs}/record.log"
ls -l "${out}"
cat "${out}"/*.log 2>/dev/null || true
uv run --no-sync python -c 'import json, sys
d = json.load(open(sys.argv[1]))
print("RESULTS", json.dumps(d.get("results")))' "${out}/mlx-metal-macos15.json"
