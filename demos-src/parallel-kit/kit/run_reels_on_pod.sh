#!/bin/bash
# On-pod helper: re-pack the kit from the synced v2 tree, arm a hardkill, and
# run job.sh for a GROUP + explicit reel list with NO_TERMINATE (pod stays up
# for inspection). Used for the staged dress-rehearsal.
# Usage: run_reels_on_pod.sh <POD_ID> <GROUP> <reel>...
set -e
POD_ID="$1"; GROUP="$2"; shift 2
REELS="$*"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq tmux zstd unzip >/dev/null 2>&1
# overlay synced scripts + tapes into the packed kit, re-tar
cp /workspace/v2/kit/*.sh /workspace/v2/kit/*.py /workspace/kit/
cp /workspace/v2/reels.yaml /workspace/kit/
rm -rf /workspace/kit/tapes && mkdir -p /workspace/kit/tapes && cp /workspace/v2/tapes/generated/*.tape /workspace/kit/tapes/
tar -czf /workspace/golden/kit.tar.gz -C /workspace kit
( cd /workspace/golden && find debs -name '*.deb' > /tmp/sl; find bin -type f >> /tmp/sl; ls fonts.tar *.tar.zst kit.tar.gz transfer.json >> /tmp/sl 2>/dev/null; xargs sha256sum < /tmp/sl > SHA256SUMS )
echo REPACK_OK
cat > /root/hardkill.sh <<EOF
#!/bin/bash
sleep 7200
curl -s -X POST "https://api.runpod.io/graphql?api_key=\$(cat /root/.runpod_key)" -H "User-Agent: Mozilla/5.0" -H "Content-Type: application/json" -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$POD_ID\\\"}) }\"}"
EOF
chmod +x /root/hardkill.sh
setsid bash /root/hardkill.sh >/root/hardkill.log 2>&1 </dev/null & disown
echo HARDKILL_120m
rm -rf /root/kit && tar -xzf /workspace/golden/kit.tar.gz -C /root
for r in $REELS; do rm -rf /workspace/reels-out/"$r"; done
RUNPOD_POD_ID="$POD_ID" NO_TERMINATE=1 nohup bash /root/kit/job.sh "$GROUP" $REELS > /workspace/qa/rehearse-$GROUP.log 2>&1 & sleep 3
pgrep -f "[j]ob.sh" >/dev/null && echo JOB_STARTED || echo JOB_FAILED
