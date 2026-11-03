#!/bin/sh
B=/home/tobias/f5b
L=$B/lem/lemonade-embeddable-2026.44.1-ubuntu-x64
O=$B/out/p5.txt
: > $O
say() { echo "--- $*" >> $O; }
setsid env HF_HOME=$B/hf $L/lemond --port 18170 --host 127.0.0.1 --no-broadcast $B/roles/chat/cache $B/roles/chat/config > $B/out/role-chat5.log 2>&1 &
sleep 6
P=$(pgrep -f 'lemond --port 18170')
curl -s -m 120 http://127.0.0.1:18170/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"lilbee-chat","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' | cut -c1-80 >> $O; echo >> $O
say "pid ppid pgid sid args (lemond started in its own session, as llama-swap would)"
ps -o pid,ppid,pgid,sid,args -e | grep -E "f5b/(lem|ownbin)" | grep -v grep | cut -c1-170 >> $O
say "embed ctx: load embed with ctx_size 1024 via recipe options"
printf '{"extra.lilbee-chat":{"ctx_size":2048,"llamacpp_args":"--parallel 2"}}' > /dev/null
say "SIGKILL lemond only"
kill -9 $P; sleep 2
ps -o pid,ppid,pgid,sid,args -e | grep -E "f5b/(lem|ownbin)" | grep -v grep | cut -c1-170 >> $O
say "killpg of the old group"
G=$(ps -o pgid= -p $(pgrep -f 'f5b/ownbin/llama-server' | head -1) | tr -d ' ')
kill -TERM -- -$G; sleep 2
pgrep -af 'f5b/(lem|ownbin)' | grep -v pgrep | cut -c1-120 >> $O; echo "(end)" >> $O
