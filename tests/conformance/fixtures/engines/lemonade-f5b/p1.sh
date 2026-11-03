#!/bin/sh
B=/home/tobias/f5b
L=$B/lem/lemonade-embeddable-2026.44.1-ubuntu-x64
M=/home/tobias/spike-1181/models
mkdir -p $B/view/chat $B/view/embeddings $B/view/reranking $B/data/cache $B/data/config $B/hf $B/out
ln -sf $M/qwen2.5-0.5b-instruct-q4_k_m.gguf $B/view/chat/lilbee-chat.gguf
ln -sf $M/bge-m3-Q4_K_M.gguf $B/view/embeddings/lilbee-embed.gguf
ln -sf $M/bge-reranker-v2-m3-Q4_K_M.gguf $B/view/reranking/lilbee-rerank.gguf
printf '{"extra_models_dir":"%s","llamacpp":{"backend":"cpu"}}' $B/view > $B/data/config/config.json
O=$B/out/p1.txt
: > $O
say() { echo "--- $*" >> $O; }
say "ss before start"; ss -ltnp 2>/dev/null | grep -E ':(8000|8001|8002|8003|9000|1815[0-9]|1816[0-9]) ' >> $O
HF_HOME=$B/hf $L/lemond --port 18160 --host 127.0.0.1 --no-broadcast $B/data/cache $B/data/config > $B/out/lemond1.log 2>&1 &
P=$!
sleep 6
say "ss after start (lemond pid $P)"; ss -ltnp 2>/dev/null | grep -E "pid=$P," >> $O
say "GET /v1/models (downloaded only, id+labels+source)"
curl -s "http://127.0.0.1:18160/v1/models" | python3 -c 'import sys,json; d=json.load(sys.stdin); [print(m["id"],m.get("labels"),m.get("source"),m.get("downloaded")) for m in d["data"] if m.get("downloaded") or m.get("source")]' >> $O
for n in lilbee-chat lilbee-embed lilbee-rerank; do
  say "POST /v1/load $n"
  curl -s -m 300 -X POST http://127.0.0.1:18160/v1/load -H 'Content-Type: application/json' -d "{\"model_name\":\"$n\",\"ctx_size\":2048}" >> $O; echo >> $O
  say "ss after load $n"; ss -ltnp 2>/dev/null | grep -E "pid=($P|$(pgrep -P $P | tr '\n' '|' | sed 's/|$//'))," >> $O
  say "tree after load $n"; ps -o pid,ppid,args --forest -g $(ps -o sid= -p $P) 2>/dev/null | cut -c1-260 >> $O
done
say "GET /v1/health"; curl -s http://127.0.0.1:18160/v1/health >> $O; echo >> $O
say "chat"; curl -s -m 120 http://127.0.0.1:18160/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"lilbee-chat","messages":[{"role":"user","content":"Capital of Italy? One word."}],"max_tokens":16}' >> $O; echo >> $O
say "embed"; curl -s -m 120 http://127.0.0.1:18160/v1/embeddings -H 'Content-Type: application/json' -d '{"model":"lilbee-embed","input":"hello"}' | python3 -c 'import sys,json; d=json.load(sys.stdin); print("dims",len(d["data"][0]["embedding"]))' >> $O
say "rerank"; curl -s -m 120 http://127.0.0.1:18160/v1/reranking -H 'Content-Type: application/json' -d '{"model":"lilbee-rerank","query":"capital of Italy","documents":["Paris is in France","Rome is in Italy"]}' >> $O; echo >> $O
say "kill lemond"; kill -TERM $P; sleep 6; pgrep -af 'lemond|llama-server' | grep f5b >> $O; echo "left: done" >> $O
