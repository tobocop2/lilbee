#!/bin/sh
B=/home/tobias/f5b
L=$B/lem/lemonade-embeddable-2026.44.1-ubuntu-x64
O=$B/out/p3.txt
: > $O
say() { echo "--- $*" >> $O; }
U=http://127.0.0.1:18160
ssfilter() { ss -ltnp 2>/dev/null | grep -E ':(800[0-9]|9000|900[0-9]|1816[0-9]) ' >> $O; }
tree() { ps -o pid,ppid,args --ppid $P | cut -c1-210 >> $O; }
printf '{"extra_models_dir":"%s","llamacpp":{"backend":"cpu"}}' $B/view > $B/data/config/config.json
HF_HOME=$B/hf $L/lemond --port 18160 --host 127.0.0.1 --no-broadcast $B/data/cache $B/data/config > $B/out/lemond4.log 2>&1 &
P=$!
sleep 6
say "G: GET /live and /v1/health before any load"
curl -s -o /dev/null -w "live=%{http_code}\n" $U/live >> $O
curl -s $U/v1/health | cut -c1-300 >> $O; echo >> $O
say "H: no explicit load: chat request auto-loads"
curl -s -m 120 $U/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"lilbee-chat","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' | cut -c1-200 >> $O; echo >> $O
tree
say "I: load lilbee-chat2 (second chat model): does it evict lilbee-chat?"
curl -s -m 300 -X POST $U/v1/load -H 'Content-Type: application/json' -d '{"model_name":"lilbee-chat2","ctx_size":2048}' >> $O; echo >> $O
tree; ssfilter
say "J: load embed + rerank, then /v1/unload lilbee-embed"
curl -s -m 300 -X POST $U/v1/load -H 'Content-Type: application/json' -d '{"model_name":"lilbee-embed"}' >> $O; echo >> $O
curl -s -m 300 -X POST $U/v1/load -H 'Content-Type: application/json' -d '{"model_name":"lilbee-rerank"}' >> $O; echo >> $O
curl -s -X POST $U/v1/unload -H 'Content-Type: application/json' -d '{"model_name":"lilbee-embed"}' >> $O; echo >> $O
tree
say "E: kill -9 one llama-server child; next request"
CP=$(pgrep -P $P -f lilbee-rerank | head -1); kill -9 $CP; sleep 2
curl -s -m 60 $U/v1/reranking -H 'Content-Type: application/json' -d '{"model":"lilbee-rerank","query":"x","documents":["a","b"]}' | cut -c1-200 >> $O; echo >> $O
tree
say "K: change extra_models_dir at runtime via /internal/set to an empty folder"
mkdir -p $B/empty
curl -s -X POST $U/internal/set -H 'Content-Type: application/json' -d "{\"extra_models_dir\":\"$B/empty\"}" | cut -c1-300 >> $O; echo >> $O
curl -s $U/v1/models | python3 -c 'import sys,json; d=json.load(sys.stdin); print([m["id"] for m in d["data"] if "lilbee" in m["id"]])' >> $O
tree
say "F: kill -9 lemond; children left?"
kill -9 $P; sleep 3
pgrep -af 'f5b/(lem|data)' | grep -v pgrep | cut -c1-200 >> $O
echo "(end F)" >> $O
for q in $(pgrep -f 'f5b/data/cache/bin/llamacpp'); do kill -TERM $q; done
sleep 2; pgrep -af 'f5b/(lem|data)' | grep -v pgrep | cut -c1-200 >> $O; echo "(end cleanup)" >> $O
