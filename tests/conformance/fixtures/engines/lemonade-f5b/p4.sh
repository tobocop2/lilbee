#!/bin/sh
B=/home/tobias/f5b
L=$B/lem/lemonade-embeddable-2026.44.1-ubuntu-x64
M=/home/tobias/spike-1181/models
O=$B/out/p4.txt
: > $O
say() { echo "--- $*" >> $O; }
mkdir -p $B/ownbin $B/roles
cp -r $B/data/cache/bin/llamacpp/cpu/. $B/ownbin/
say "ownbin"; ls $B/ownbin | head -20 >> $O
i=0
for r in chat embed rerank; do
  case $r in chat) mode=chat; f=qwen2.5-0.5b-instruct-q4_k_m.gguf; extra='"ctx_size":2048,"llamacpp_args":"--parallel 2"';;
    embed) mode=embeddings; f=bge-m3-Q4_K_M.gguf; extra='"ctx_size":4096,"llamacpp_args":"--parallel 3"';;
    rerank) mode=reranking; f=bge-reranker-v2-m3-Q4_K_M.gguf; extra='"ctx_size":2048';; esac
  mkdir -p $B/roles/$r/cache $B/roles/$r/config $B/roles/$r/view/$mode
  ln -sf $M/$f $B/roles/$r/view/$mode/lilbee-$r.gguf
  printf '{"extra_models_dir":"%s","llamacpp":{"backend":"cpu","cpu_bin":"%s"}}' $B/roles/$r/view $B/ownbin > $B/roles/$r/config/config.json
  printf '{"extra.lilbee-%s":{%s}}' $r "$extra" > $B/roles/$r/config/recipe_options.json
done
say "ss before"; ss -ltnp 2>/dev/null | grep -E ':(800[0-9]|900[0-9]|1817[0-9]) ' >> $O
PIDS=""
n=0
for r in chat embed rerank; do
  HF_HOME=$B/hf $L/lemond --port 1817$n --host 127.0.0.1 --no-broadcast $B/roles/$r/cache $B/roles/$r/config > $B/out/role-$r.log 2>&1 &
  eval "P_$r=$!"; PIDS="$PIDS $!"; n=$((n+1))
done
sleep 7
say "ss after 3 lemonds start (no model loaded)"; ss -ltnp 2>/dev/null | grep -E ':(800[0-9]|900[0-9]|1817[0-9]) ' >> $O
say "first request per role, using the Lemonade model name"
curl -s -m 120 http://127.0.0.1:18170/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"lilbee-chat","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' | cut -c1-160 >> $O; echo >> $O
curl -s -m 120 http://127.0.0.1:18171/v1/embeddings -H 'Content-Type: application/json' -d '{"model":"lilbee-embed","input":"hello"}' | python3 -c 'import sys,json; d=json.load(sys.stdin); print("dims",len(d["data"][0]["embedding"]))' >> $O
curl -s -m 120 http://127.0.0.1:18172/v1/reranking -H 'Content-Type: application/json' -d '{"model":"lilbee-rerank","query":"capital of Italy","documents":["Paris is in France","Rome is in Italy"]}' | cut -c1-200 >> $O; echo >> $O
say "wrong model name (the llama-swap id) sent to a role server"
curl -s -m 20 http://127.0.0.1:18170/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"chat-0","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' | cut -c1-260 >> $O; echo >> $O
say "ss after requests"; ss -ltnp 2>/dev/null | grep -E ':(800[0-9]|900[0-9]|1817[0-9]) ' >> $O
say "process tree"; ps -o pid,ppid,args -e --forest 2>/dev/null | grep -E 'f5b/(lem|roles|ownbin|data)' | grep -v grep | cut -c1-250 >> $O
say "downloads? cache/bin per role and log lines"; ls $B/roles/*/cache/bin 2>&1 >> $O; grep -h -i -E "install|download" $B/out/role-*.log | head -5 >> $O
say "SIGTERM all three lemonds"
for p in $PIDS; do kill -TERM $p; done
sleep 12
pgrep -af 'f5b/(lem|roles|ownbin|data)' | grep -v pgrep | cut -c1-200 >> $O; echo "(end)" >> $O
