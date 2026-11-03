#!/bin/sh
B=/home/tobias/f5b
L=$B/lem/lemonade-embeddable-2026.44.1-ubuntu-x64
O=$B/out/p2.txt
: > $O
say() { echo "--- $*" >> $O; }
U=http://127.0.0.1:18160
ssfilter() { ss -ltnp 2>/dev/null | grep -E ':(8000|8001|8002|8003|8004|9000|1816[0-9]) ' >> $O; }
# A: hold 8001 and 9000 with plain listeners, then start and load chat and embed
python3 -c '
import socket,time,sys
ss=[]
for p in (8001,9000):
    s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(("127.0.0.1",p)); s.listen(1); ss.append(s)
time.sleep(60)
' &
D=$!
sleep 1
say "A: dummy listeners hold 8001 and 9000 (pid $D)"; ssfilter
HF_HOME=$B/hf $L/lemond --port 18160 --host 127.0.0.1 --no-broadcast $B/data/cache $B/data/config > $B/out/lemond2.log 2>&1 &
P=$!
sleep 6
say "A: after start"; ssfilter
say "A: health websocket_port"; curl -s $U/v1/health | python3 -c 'import sys,json; print(json.load(sys.stdin).get("websocket_port"))' >> $O
say "A: load chat"; curl -s -m 300 -X POST $U/v1/load -H 'Content-Type: application/json' -d '{"model_name":"lilbee-chat","ctx_size":2048}' >> $O; echo >> $O
say "A: after load chat"; ssfilter
say "A: load embed"; curl -s -m 300 -X POST $U/v1/load -H 'Content-Type: application/json' -d '{"model_name":"lilbee-embed"}' >> $O; echo >> $O
say "A: after load embed"; ssfilter
say "A: llama-server args"; ps -o args= --ppid $P | cut -c1-200 >> $O
kill -TERM $P; kill $D; sleep 12
say "A: left"; pgrep -af 'f5b/(lem|data)' | grep -v pgrep >> $O
# B: websocket_port override in config.json
printf '{"extra_models_dir":"%s","websocket_port":18161,"llamacpp":{"backend":"cpu"}}' $B/view > $B/data/config/config.json
HF_HOME=$B/hf $L/lemond --port 18160 --host 127.0.0.1 --no-broadcast $B/data/cache $B/data/config > $B/out/lemond3.log 2>&1 &
P=$!
sleep 6
say "B: websocket_port=18161 in config; listeners"; ssfilter
# C: register by path; try variants
say "C1: register checkpoint=abs path recipe llamacpp"
curl -s -X POST $U/v1/models/register -H 'Content-Type: application/json' -d '{"model_name":"user.pathchat","recipe":"llamacpp","checkpoint":"/home/tobias/spike-1181/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"}' >> $O; echo >> $O
say "C2: register source=local"
curl -s -X POST $U/v1/models/register -H 'Content-Type: application/json' -d '{"model_name":"user.pathchat2","recipe":"llamacpp","source":"local","checkpoint":"/home/tobias/spike-1181/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"}' >> $O; echo >> $O
say "C3: list user models + user_models.json"
curl -s $U/v1/models | python3 -c 'import sys,json; d=json.load(sys.stdin); [print(m["id"],m.get("downloaded"),m.get("labels"),m.get("checkpoint")) for m in d["data"] if "path" in m["id"]]' >> $O
cat $B/data/config/user_models.json >> $O 2>&1; echo >> $O
say "C4: load user.pathchat"
curl -s -m 300 -X POST $U/v1/load -H 'Content-Type: application/json' -d '{"model_name":"user.pathchat","ctx_size":2048}' | cut -c1-400 >> $O; echo >> $O
# D: extra_models_dir added at runtime: new symlink then rescan
ln -sf /home/tobias/spike-1181/models/qwen2.5-1.5b-instruct-q4_k_m.gguf $B/view/chat/lilbee-chat2.gguf
sleep 4
say "D: runtime new symlink visible?"
curl -s $U/v1/models | python3 -c 'import sys,json; d=json.load(sys.stdin); [print(m["id"],m.get("downloaded")) for m in d["data"] if "lilbee" in m["id"]]' >> $O
say "D: ss + tree after path load"; ssfilter; ps -o pid,ppid,args --ppid $P | cut -c1-220 >> $O
say "E: kill -9 llama-server child; does lemond respawn on next request?"
CP=$(pgrep -P $P | head -1); kill -9 $CP; sleep 2
curl -s -m 60 $U/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"user.pathchat","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' | cut -c1-300 >> $O; echo >> $O
ps -o pid,ppid,args --ppid $P | cut -c1-200 >> $O
say "F: kill -9 lemond; children?"
kill -9 $P; sleep 3
pgrep -af 'f5b/(lem|data)' | grep -v pgrep >> $O
echo "(end F)" >> $O
