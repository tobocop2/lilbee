#!/usr/bin/env bash
# Drive a full MS MARCO ingest on the native per-GPU path, from a laptop.
#
#   ./pod_native.sh up        pick a datacenter with capacity, make the volume,
#                             provision, launch the ingest, open the dashboard
#   ./pod_native.sh attach    re-open the local dashboard from any terminal
#   ./pod_native.sh watch     ssh in and attach the SAME dashboard on the pod
#   ./pod_native.sh status    one-shot summary, no tmux
#   ./pod_native.sh log       follow the run log
#   ./pod_native.sh publish   swap the GPU pod for a cheap CPU pod on the SAME
#                             volume and upload everything from there
#   ./pod_native.sh resume    provision a REPLACEMENT pod on the same volume and
#                             continue the ingest where it stopped
#   ./pod_native.sh down      delete the pod, KEEP the volume (and the index)
#   ./pod_native.sh nuke      delete the pod AND the volume. Unrecoverable.
#
# THE VOLUME IS THE POINT. The previous full run finished and was lost because
# the index was on the container disk: the pod was stopped, RunPod reallocated
# the GPUs, the pod could not restart, and six hours of work went with it.
# Stopping a pod does not reserve its GPUs. Here the index lives on a network
# volume, so 'resume' is a NEW pod attached to the same data, and it is allowed
# to land on different hardware, in fact on any host in that datacenter.
#
# The cost of that is a datacenter pin: a volume cannot move. 'up' therefore
# picks the datacenter by 8-pack availability at that moment and creates the
# volume there, rather than the other way round.
#
# The dashboard and the recording run HERE, on your machine, over ssh reads.
# They cost the pod no CPU, no GPU and no disk, and the recording keeps your own
# tmux configuration because it is your tmux.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
STATE_DIR="$HOME/.msmarco9m"; STATE="$STATE_DIR/run.json"
KEY="$HOME/.ssh/runpod_qa"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o IdentitiesOnly=yes -o ServerAliveInterval=30 -i "$KEY")

: "${IMAGE:=runpod/pytorch:1.0.7-rc.138-cu1281-torch271-ubuntu2404}"
: "${CPU_IMAGE:=runpod/base:1.0.2-ubuntu2204}"
: "${GPU_NAME:=NVIDIA H100 80GB HBM3}"
: "${GPUS:=8}"
# Container disk holds the corpus only. 8.8M files of ~325 bytes still occupy a
# 4KB block each, so the corpus is ~36GB on disk, not 3GB.
: "${DISK:=120}"
# Volume holds the merged index (~144GB of vectors) AND the shards, which are
# kept as the resume state and hold a second copy of every vector.
: "${VOL_GB:=500}"
: "${HOURS:=10}"
: "${HF_REPO:=beeberg/msmarco-ingest-checkpoint}"

die()  { echo "error: $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null || die "$1 not on PATH"; }

PY_STATE='
import json, pathlib, sys
p = pathlib.Path.home() / ".msmarco9m" / "run.json"
d = json.loads(p.read_text()) if p.exists() else {}
if len(sys.argv) > 1:
    d.update(json.loads(sys.argv[1]))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1))
print(json.dumps(d))'
save() { python3 -c "$PY_STATE" "$1" >/dev/null; }
load() {
  [ -f "$STATE" ] || die "no run recorded. './pod_native.sh up' first"
  eval "$(python3 -c "
import json, pathlib
d = json.loads(pathlib.Path('$STATE').read_text())
for k in ('pod', 'host', 'port', 'volume', 'dc'):
    print(f'{k.upper()}={d.get(k, \"\")!r}')")"
  [ -n "${POD:-}" ] || die "state has no pod: $STATE"
}
pod_ssh() { ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@"; }
api_key() { python3 -c "
import pathlib, tomllib
c = tomllib.loads(pathlib.Path.home().joinpath('.runpod/config.toml').read_text())
print(c.get('apikey') or c['default']['api_key'])"; }

# Datacenters that can serve this pod's GPU count RIGHT NOW. A network volume
# pins the run to one of them for its whole life, so the choice is made against
# live capacity rather than a preference.
pick_dc() {
  # The candidate list comes from the CLI, not GraphQL: the schema has no
  # dataCenters root field, and asking for one returns a bare HTTP 400 that reads
  # like a credentials problem.
  runpodctl datacenter list 2>/dev/null > /tmp/pod_native.dc.json || return 1
  python3 - "$(api_key)" "$GPU_NAME" "$GPUS" /tmp/pod_native.dc.json <<'PY'
import json, sys, urllib.request
key, gpu, count, dcfile = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]


def gql(q):
    req = urllib.request.Request(
        f"https://api.runpod.io/graphql?api_key={key}",
        data=json.dumps({"query": q}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "pod_native"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


dcs = [
    dc["id"]
    for dc in json.load(open(dcfile))
    if any(g.get("gpuId") == gpu for g in dc.get("gpuAvailability") or [])
]
ok = []
for dc in dcs:
    q = ('query{gpuTypes(input:{id:"%s"}){lowestPrice(input:{gpuCount:%d,dataCenterId:"%s"})'
         '{uninterruptablePrice stockStatus}}}' % (gpu, count, dc))
    try:
        price = (gql(q)["data"]["gpuTypes"][0] or {}).get("lowestPrice") or {}
    except Exception:
        continue
    # A null price means this datacenter cannot assemble the pack at all, which
    # is a harder no than a "Low" stock status.
    if price.get("uninterruptablePrice"):
        ok.append((dc, price["uninterruptablePrice"], price.get("stockStatus") or "?"))
for dc, price, stock in ok:
    print(f"{dc} {price} {stock}", file=sys.stderr)
print(ok[0][0] if ok else "")
PY
}

endpoint() {  # $1 = pod id; prints "host port" once ssh is exposed
  python3 - "$(api_key)" "$1" <<'PY'
import json, sys, time, urllib.request
key, pod = sys.argv[1], sys.argv[2]
q = 'query{pod(input:{podId:"%s"}){runtime{ports{ip isIpPublic privatePort publicPort}}}}' % pod
for _ in range(90):
    req = urllib.request.Request(
        f"https://api.runpod.io/graphql?api_key={key}",
        data=json.dumps({"query": q}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "pod_native"},
    )
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except Exception:
        time.sleep(10)
        continue
    # runtime is present-but-null while the pod boots, so every level needs `or {}`.
    ports = (((d.get("data") or {}).get("pod") or {}).get("runtime") or {}).get("ports") or []
    for p in ports:
        if p.get("privatePort") == 22 and p.get("isIpPublic"):
            print(p["ip"], p["publicPort"])
            sys.exit(0)
    time.sleep(10)
sys.exit(1)
PY
}

wait_ssh() {
  echo "waiting for sshd..."
  for _ in $(seq 1 40); do pod_ssh true 2>/dev/null && return 0; sleep 10; done
  return 1
}

upload_scripts() {
  scp "${SSH_OPTS[@]}" -P "$PORT" \
    "$HERE/native9m.sh" "$HERE/publish9m.sh" "$HERE/idlewatch.sh" "$HERE/dash.sh" \
    "$HERE/rows_sampler.py" "$HERE/extract_hist.py" "$HERE/summarize.py" \
    "root@$HOST:/root/" >/dev/null || die "scp scripts"
  pod_ssh 'chmod +x /root/native9m.sh /root/publish9m.sh /root/idlewatch.sh /root/dash.sh
    command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux) >/dev/null 2>&1
    tmux -V'
}

# Provision a pod attached to $VOLUME in $DC, retrying: multi-GPU capacity flaps
# and a create attempt costs nothing.
provision() {  # $1 = name, $2 = gpu count (0 = cpu pod); sets POD/HOST/PORT
  local name="$1" gpus="$2" at pod="" args=()
  at=$(python3 -c "
from datetime import datetime, timedelta, UTC
print((datetime.now(UTC) + timedelta(hours=$HOURS)).strftime('%Y-%m-%dT%H:%M:%SZ'))")
  if [ "$gpus" = "0" ]; then
    args=(--compute-type cpu --image "$CPU_IMAGE" --container-disk-in-gb 60)
  else
    args=(--image "$IMAGE" --gpu-id "$GPU_NAME" --gpu-count "$gpus"
          --container-disk-in-gb "$DISK")
  fi
  for attempt in $(seq 1 "${ATTEMPTS:-90}"); do
    pod=$(runpodctl pod create --name "$name" "${args[@]}" \
            --network-volume-id "$VOLUME" --data-center-ids "$DC" \
            --terminate-after "$at" --ports 22/tcp 2>/dev/null \
          | sed -n 's/.*"id": *"\([^"]*\)".*/\1/p' | head -1)
    [ -n "$pod" ] && break
    echo "  attempt $attempt: no capacity in $DC, retrying"; sleep 20
  done
  [ -n "$pod" ] || return 1
  POD="$pod"; echo "provisioned $POD"
  local ep; ep=$(endpoint "$POD") || { runpodctl pod delete "$POD" >/dev/null 2>&1; return 1; }
  HOST=${ep% *}; PORT=${ep#* }
  save "{\"pod\": \"$POD\", \"host\": \"$HOST\", \"port\": $PORT}"
  echo "ssh root@$HOST:$PORT"
  wait_ssh || return 1
}

launch_ingest() {
  local token; token=$(cat "$HOME/.cache/huggingface/token")
  # `env` prefix, not a bare `${VAR:+NAME=value}`: the shell decides what is an
  # assignment BEFORE expanding, so an expanded assignment runs as a command and
  # exits 127. That silently killed two runs of an earlier harness.
  # Launch a private SNAPSHOT of the script, never /root/native9m.sh itself.
  # bash reads a script incrementally by byte offset, so overwriting one while it
  # runs makes it resume mid-token: uploading a fix mid-run once killed a
  # completed ingest's post-merge steps with "n_main,: command not found".
  # Re-uploading now only affects the NEXT launch.
  pod_ssh "mkdir -p /root/run && cp /root/native9m.sh /root/run/native9m.\$\$.sh" || die "snapshot"
  pod_ssh "rm -f /workspace/FAILED_AT /workspace/MERGE_DONE /workspace/COUNT_MISMATCH; \
    RUN=\$(ls -t /root/run/native9m.*.sh | head -1); \
    setsid env HF_TOKEN='$token' HF_REPO='$HF_REPO' \
    ${EXTRACT_GLOB:+EXTRACT_GLOB='$EXTRACT_GLOB'} ${ANN:+ANN='$ANN'} \
    ${TRACE:+TRACE='$TRACE'} ${PROFILE:+PROFILE='$PROFILE'} \
    ${EMBED_MODEL:+EMBED_MODEL='$EMBED_MODEL'} ${EMBED_DIM:+EMBED_DIM='$EMBED_DIM'} \
    ${CORPUS_URL:+CORPUS_URL='$CORPUS_URL'} ${BRANCH:+BRANCH='$BRANCH'} \
    bash \"\$RUN\" </dev/null >/dev/null 2>&1 & echo launched" \
    || die "launch failed"
  # The watchdog stops the pod when the run is over or has died, so an overnight
  # run cannot bill through the morning. It needs the pod's own id and an API key.
  pod_ssh "mkdir -p /workspace/status && echo '$POD' > /workspace/status/pod_id; \
    command -v runpodctl >/dev/null || { curl -fsSL -o /usr/local/bin/runpodctl \
      https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 \
      && chmod +x /usr/local/bin/runpodctl; }; \
    mkdir -p /root/.runpod && chmod 700 /root/.runpod && \
    printf 'apikey = \"%s\"\n' '$(api_key)' > /root/.runpod/config.toml && chmod 600 /root/.runpod/config.toml"
  pod_ssh "STATE_DIR=/workspace ${GRACE_MIN:+GRACE_MIN=$GRACE_MIN} ${IDLE_MIN:+IDLE_MIN=$IDLE_MIN} \
    setsid bash /root/idlewatch.sh </dev/null >/dev/null 2>&1 & echo armed" || true
}

cmd_up() {
  have runpodctl; have ssh; have scp; have python3
  [ -f "$KEY" ] || die "missing ssh key $KEY"
  [ -f "$HOME/.cache/huggingface/token" ] || die "missing HF token"
  [ -f "$STATE" ] && die "a run is already recorded. 'attach', 'down' or 'nuke' first"

  echo "looking for a datacenter that can serve ${GPUS}x${GPU_NAME}..."
  DC=$(pick_dc)
  [ -n "$DC" ] || die "no datacenter can assemble ${GPUS}x${GPU_NAME} right now"
  echo "chosen: $DC"

  echo "creating a ${VOL_GB}GB network volume in $DC"
  VOLUME=$(runpodctl network-volume create --name "msmarco9m" --size "$VOL_GB" \
             --data-center-id "$DC" 2>/dev/null \
           | sed -n 's/.*"id": *"\([^"]*\)".*/\1/p' | head -1)
  [ -n "$VOLUME" ] || die "volume create failed in $DC"
  save "{\"volume\": \"$VOLUME\", \"dc\": \"$DC\"}"
  echo "volume $VOLUME (kept when the pod goes; 'nuke' deletes it)"

  provision "msmarco9m" "$GPUS" || die "never provisioned"
  upload_scripts
  launch_ingest
  start_watching
}

# The dashboard and the recorder are started detached rather than attached to,
# so 'up' returns and the recording begins whether or not anyone is looking.
start_watching() {
  sleep 5
  EXPECTED="${EXPECTED:-8841823}" "$HERE/dash.sh" start
  [ "${RECORD:-1}" = "1" ] && "$HERE/rec.sh" start
  echo
  echo "  watch it here:      ./pod_native.sh attach"
  echo "  watch it on the pod: ./pod_native.sh watch"
  echo "  one-shot summary:    ./pod_native.sh status"
}

cmd_resume() {
  load
  echo "replacement pod on volume $VOLUME in $DC (the index is already there)"
  runpodctl pod delete "$POD" >/dev/null 2>&1
  provision "msmarco9m-resume" "$GPUS" || die "never provisioned"
  upload_scripts
  launch_ingest
  start_watching
}

cmd_publish() {
  load
  local done_at
  done_at=$(pod_ssh 'cat /workspace/MERGE_DONE 2>/dev/null' 2>/dev/null)
  [ -n "$done_at" ] || echo "warning: MERGE_DONE is not set; publishing whatever is on the volume"
  echo "releasing the GPU pod: an upload is network-bound and does not need eight H100s"
  runpodctl pod delete "$POD" >/dev/null 2>&1
  sleep 5
  if ATTEMPTS=20 provision "msmarco9m-publish" 0; then
    echo "publishing from a CPU pod"
  else
    echo "no CPU pod in $DC; falling back to one cheap GPU pod"
    GPU_NAME="NVIDIA GeForce RTX 4090" ATTEMPTS=20 provision "msmarco9m-publish" 1 \
      || die "no pod available to publish from"
  fi
  upload_scripts
  local token; token=$(cat "$HOME/.cache/huggingface/token")
  echo "uploading (telemetry, then dataset, then the index)"
  pod_ssh "HF_TOKEN='$token' HF_REPO='$HF_REPO' ${UPLOAD_INDEX:+UPLOAD_INDEX=$UPLOAD_INDEX} \
    setsid bash /root/publish9m.sh </dev/null >> /workspace/publish.log 2>&1 & echo started"
  echo "follow it with: ./pod_native.sh log publish"
}

cmd_status() {
  load
  echo "pod $POD  volume $VOLUME  dc $DC  ssh root@$HOST:$PORT"
  pod_ssh 'echo "  phase: $(cat /workspace/status/phase 2>/dev/null || echo "?")"
    sed "s/^/  /" /workspace/status/run.env 2>/dev/null
    sed "s/^/  /" /workspace/status/counts 2>/dev/null
    tail -3 /workspace/prof/rows.csv 2>/dev/null | sed "s/^/  rows /"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed "s/^/  card /"
    df -h /workspace | tail -1 | sed "s/^/  vol /"
    for f in MERGE_DONE PUBLISH_DONE FAILED_AT COUNT_MISMATCH; do
      [ -e "/workspace/$f" ] && echo "  $f: $(cat /workspace/$f)"
    done' 2>/dev/null
}

cmd_log() {
  load
  case "${2:-ingest}" in
    publish) pod_ssh 'tail -F /workspace/publish.log' ;;
    sync)    pod_ssh "tail -F /workspace/sync.out | tr '\r' '\n'" ;;
    *)       pod_ssh 'tail -F /workspace/ingest.log' ;;
  esac
}

cmd_down() {
  load
  echo "deleting pod $POD; volume $VOLUME KEEPS the index"
  runpodctl pod delete "$POD" >/dev/null 2>&1
  save '{"pod": ""}'
  echo "done. 'resume' brings up a new pod on the same volume; 'nuke' deletes the data."
}

cmd_nuke() {
  load
  echo "this DELETES the index. pod $POD and volume $VOLUME."
  printf 'type the volume id to confirm: '
  read -r answer
  [ "$answer" = "$VOLUME" ] || die "not confirmed"
  runpodctl pod delete "$POD" >/dev/null 2>&1
  sleep 5
  runpodctl network-volume delete "$VOLUME" >/dev/null 2>&1
  rm -f "$STATE"
  echo "deleted."
}

# The same dashboard, served from the pod. For watching from a machine that is
# not the one recording, or from a phone: ssh in and the panes are already there.
cmd_watch() {
  load
  ssh -t "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" 'bash /root/dash.sh serve'
}

case "${1:-}" in
  dc) pick_dc >/dev/null ;;
  up) cmd_up ;;
  attach) exec "$HERE/dash.sh" attach ;;
  watch) cmd_watch ;;
  resume) cmd_resume ;;
  publish) cmd_publish ;;
  status) cmd_status ;;
  log) cmd_log "$@" ;;
  down) cmd_down ;;
  nuke) cmd_nuke ;;
  *) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
