#!/usr/bin/env bash
# Provision an 8xH100 pod, start the full 9M MS MARCO ingest on it, and drop you
# into a tmux monitoring session. Runs standalone: no agent, no session state
# beyond a small json under ~/.msmarco9m.
#
#   ./pod9m.sh up        provision, upload, launch the ingest, attach
#   ./pod9m.sh attach    re-attach from any terminal, any time
#   ./pod9m.sh status    one-shot summary printed locally, no tmux
#   ./pod9m.sh log       tail the run log locally
#   ./pod9m.sh fetch     copy logs and the run's numbers to ./results/
#   ./pod9m.sh resume    restart a stopped pod AND continue the ingest where it left off
#   ./pod9m.sh watchdog  show the idle watchdog's log
#   ./pod9m.sh down      delete the pod (ALWAYS run this when finished)
#
# Overnight-safe: a watchdog on the pod powers it off GRACE_MIN after the run
# ends, or after IDLE_MIN with no busy card and no worker (the crashed case).
# Poweroff stops GPU billing and keeps the disk, so results survive until you
# run 'resume' to fetch them or 'down' to delete. The pod's terminate-after
# (HOURS) is the hard backstop behind both.
#
# Detach from tmux with ctrl-b d. The ingest keeps running: it is started with
# setsid before tmux exists, so closing your laptop cannot kill it.
#
# Requirements: runpodctl logged in, ~/.ssh/runpod_qa, an HF token at
# ~/.cache/huggingface/token, and a lilbee wheel in PAYLOAD (see stage_payload.sh).
#
# Knobs (environment):
#   PAYLOAD    dir with the lilbee wheel + merge/manifest tools (required for up)
#   GPUS       default 8
#   DISK       container disk GB, default 500. The merge holds the shards and the
#              merged copy at once: ~144GB of vectors each at 8.8M x 4096 dims.
#   HOURS      pod terminate-after backstop, default 12
#   PLAN_POOL  per-worker planning threads; empty = cores/workers (see ingest9m.sh)
#   EXTRACT_GLOB  unpack a subset for a smaller trial, e.g. 'documents/00[0-7]*'
#   HF_REPO    where to push results, default beeberg/msmarco-ingest-checkpoint.
#              Lands as dataset/ (parquet + jsonl) and index/ (the full index)
#              subdirectories, so existing repo contents are untouched.
#   UPLOAD_INDEX  0 to push only the datasets and skip the ~150GB index
#   EXPORT     0 to skip export and upload entirely
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
STATE_DIR="$HOME/.msmarco9m"
STATE="$STATE_DIR/pod.json"
KEY="$HOME/.ssh/runpod_qa"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o IdentitiesOnly=yes -o ServerAliveInterval=30 -i "$KEY")
IMAGE="runpod/pytorch:1.0.7-rc.138-cu1281-torch271-ubuntu2404"
GPU_NAME="NVIDIA H100 80GB HBM3"
: "${GPUS:=8}"; : "${DISK:=500}"; : "${HOURS:=12}"

die() { echo "error: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null || die "$1 not on PATH"; }

read_state() {
  [ -f "$STATE" ] || die "no pod recorded. run './pod9m.sh up' first"
  POD=$(sed -n 's/.*"pod": *"\([^"]*\)".*/\1/p' "$STATE")
  HOST=$(sed -n 's/.*"host": *"\([^"]*\)".*/\1/p' "$STATE")
  PORT=$(sed -n 's/.*"port": *\([0-9]*\).*/\1/p' "$STATE")
  [ -n "${POD:-}" ] && [ -n "${HOST:-}" ] && [ -n "${PORT:-}" ] || die "state file is unreadable: $STATE"
}

pod_ssh() { ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@"; }

api_key() {
  python3 - <<'PY'
import pathlib, tomllib
cfg = tomllib.loads(pathlib.Path.home().joinpath(".runpod/config.toml").read_text())
print(cfg.get("apikey") or cfg["default"]["api_key"])
PY
}

endpoint() {  # $1 = pod id; prints "host port" once ssh is exposed
  local key; key=$(api_key)
  python3 - "$key" "$1" <<'PY'
import json, sys, time, urllib.request
key, pod = sys.argv[1], sys.argv[2]
q = '{pod(input:{podId:"%s"}){runtime{ports{ip isIpPublic privatePort publicPort}}}}' % pod
for _ in range(60):
    req = urllib.request.Request(f"https://api.runpod.io/graphql?api_key={key}",
        data=json.dumps({"query": "query" + q}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "pod9m"})
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except Exception:
        time.sleep(10); continue
    p = (((d.get("data") or {}).get("pod") or {}).get("runtime") or {}).get("ports") or []
    for e in p:
        if e.get("privatePort") == 22 and e.get("isIpPublic"):
            print(e["ip"], e["publicPort"]); sys.exit(0)
    time.sleep(10)
sys.exit(1)
PY
}

cmd_up() {
  need runpodctl; need ssh; need scp; need python3
  [ -n "${PAYLOAD:-}" ] || die "set PAYLOAD to a dir holding the lilbee wheel (see stage_payload.sh)"
  [ -d "$PAYLOAD" ] || die "PAYLOAD is not a directory: $PAYLOAD"
  ls "$PAYLOAD"/lilbee-*.whl >/dev/null 2>&1 || die "no lilbee wheel in $PAYLOAD"
  [ -f "$KEY" ] || die "missing ssh key $KEY"
  local token_file="$HOME/.cache/huggingface/token"
  [ -f "$token_file" ] || die "missing HF token at $token_file"
  if [ -f "$STATE" ]; then
    read_state
    echo "a pod is already recorded ($POD). './pod9m.sh attach' or './pod9m.sh down' first." >&2
    exit 1
  fi
  mkdir -p "$STATE_DIR"

  local at pod=""
  at=$(python3 -c "
from datetime import datetime, timedelta, UTC
print((datetime.now(UTC) + timedelta(hours=$HOURS)).strftime('%Y-%m-%dT%H:%M:%SZ'))")
  echo "provisioning ${GPUS}xH100, ${DISK}GB disk, terminate-after ${HOURS}h..."
  for attempt in $(seq 1 90); do
    pod=$(runpodctl pod create --name "msmarco9m" --image "$IMAGE" \
            --gpu-id "$GPU_NAME" --gpu-count "$GPUS" \
            --container-disk-in-gb "$DISK" --terminate-after "$at" --ports 22/tcp \
            2>/dev/null | sed -n 's/.*"id": *"\([^"]*\)".*/\1/p' | head -1)
    [ -n "$pod" ] && break
    echo "  attempt $attempt: no ${GPUS}xH100 capacity, retrying"; sleep 20
  done
  [ -n "$pod" ] || die "never provisioned"
  echo "provisioned $pod"

  local ep; ep=$(endpoint "$pod") || { runpodctl pod delete "$pod" >/dev/null 2>&1; die "no ssh endpoint"; }
  HOST=${ep% *}; PORT=${ep#* }; POD=$pod
  printf '{"pod": "%s", "host": "%s", "port": %s}\n' "$POD" "$HOST" "$PORT" > "$STATE"
  echo "ssh root@$HOST:$PORT   (recorded in $STATE)"

  echo "waiting for sshd..."
  for _ in $(seq 1 40); do pod_ssh true 2>/dev/null && break; sleep 10; done
  pod_ssh true 2>/dev/null || die "ssh never came up"

  echo "uploading scripts and payload..."
  scp "${SSH_OPTS[@]}" -P "$PORT" "$HERE/ingest9m.sh" "$HERE/monitor9m.sh" "$HERE/idlewatch.sh" "$HERE/export9m.sh" "root@$HOST:/root/" >/dev/null || die scp
  pod_ssh 'mkdir -p /root/payload' || die mkdir
  scp "${SSH_OPTS[@]}" -P "$PORT" "$PAYLOAD"/* "root@$HOST:/root/payload/" >/dev/null || die "scp payload"
  pod_ssh 'command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux) >/dev/null 2>&1; tmux -V'

  echo "starting the ingest (detached; survives your terminal)..."
  pod_ssh "chmod +x /root/ingest9m.sh /root/monitor9m.sh; rm -f /root/RUN_DONE /root/FAILED_AT; \
    HF_TOKEN=$(cat "$token_file") ${PLAN_POOL:+PLAN_POOL=$PLAN_POOL} ${EXTRACT_GLOB:+EXTRACT_GLOB='$EXTRACT_GLOB'} \
    ${HF_REPO:+HF_REPO=$HF_REPO} ${UPLOAD_INDEX:+UPLOAD_INDEX=$UPLOAD_INDEX} ${EXPORT:+EXPORT=$EXPORT} \
    WORKERS=$GPUS setsid bash /root/ingest9m.sh </dev/null >/dev/null 2>&1 & echo started" \
    || die "launch failed"
  # The watchdog stops the pod through the RunPod API, so the pod needs the key
  # and its own id. The key is mode 600 on a box you rented and dies with it, and
  # your HF token is already there; NO_SELF_STOP=1 skips it and the watchdog
  # falls back to powering the box off, which is best effort in a container.
  pod_ssh "mkdir -p /root/status && echo '$POD' > /root/status/pod_id"
  if [ "${NO_SELF_STOP:-0}" != "1" ]; then
    pod_ssh "command -v runpodctl >/dev/null || { \
        curl -fsSL -o /usr/local/bin/runpodctl \
          https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 \
        && chmod +x /usr/local/bin/runpodctl; }; runpodctl version 2>&1 | head -1" || true
    pod_ssh "mkdir -p /root/.runpod && chmod 700 /root/.runpod && \
      printf 'apikey = \"%s\"\n' '$(api_key)' > /root/.runpod/config.toml && \
      chmod 600 /root/.runpod/config.toml && echo 'self-stop enabled'"
  else
    echo "NO_SELF_STOP=1: watchdog will power off rather than stop via the API"
  fi
  echo "arming the idle watchdog (poweroff ${GRACE_MIN:-20}m after the run ends, or ${IDLE_MIN:-30}m idle)..."
  pod_ssh "chmod +x /root/idlewatch.sh; \
    ${GRACE_MIN:+GRACE_MIN=$GRACE_MIN} ${IDLE_MIN:+IDLE_MIN=$IDLE_MIN} \
    setsid bash /root/idlewatch.sh </dev/null >/dev/null 2>&1 & echo armed" || die "watchdog"
  echo
  echo "ingest running. attaching the monitor; ctrl-b d detaches without stopping it."
  sleep 3
  cmd_attach
}

cmd_attach() {
  read_state
  ssh -t "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" 'bash /root/monitor9m.sh'
}

cmd_status() {
  read_state
  echo "pod $POD  ssh root@$HOST:$PORT"
  pod_ssh 'cat /root/status/run.env 2>/dev/null | sed "s/^/  /"
    echo "  ---"
    grep -aE "^INGEST|^MERGE|^FATAL" /root/ingest.log 2>/dev/null | tail -5 | sed "s/^/  /"
    for i in $(seq 0 7); do
      [ -f /root/w$i/sync.log ] || continue
      printf "  w%s %s\n" "$i" "$(tr "\r" "\n" < /root/w$i/sync.log | grep -a "files/s" | tail -1 | grep -oE "examined [0-9]+/[0-9]+ files \([0-9]+%, [0-9]+ files/s")"
    done
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed "s/^/  card /"
    [ -f /root/RUN_DONE ] && echo "  RUN_DONE"
    [ -f /root/FAILED_AT ] && echo "  *** FAILED ***"' 2>/dev/null
}

cmd_log() { read_state; pod_ssh 'tail -F /root/ingest.log'; }

cmd_fetch() {
  read_state
  local out="$HERE/results/9m-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$out"
  scp "${SSH_OPTS[@]}" -P "$PORT" "root@$HOST:/root/ingest.log" "$out/" 2>/dev/null
  for i in $(seq 0 $((GPUS-1))); do
    scp "${SSH_OPTS[@]}" -P "$PORT" "root@$HOST:/root/w$i/sync.log" "$out/w$i.sync.log" 2>/dev/null
  done
  pod_ssh 'grep -aE "^INGEST|^MERGE|merged |^FATAL" /root/ingest.log' > "$out/summary.txt" 2>/dev/null
  echo "fetched to $out"
}

cmd_resume() {
  read_state
  echo "starting pod $POD (it powers itself off when idle; the disk survived)"
  runpodctl start pod "$POD" >/dev/null 2>&1
  local ep; ep=$(endpoint "$POD") || die "pod did not come back"
  HOST=${ep% *}; PORT=${ep#* }
  printf '{"pod": "%s", "host": "%s", "port": %s}\n' "$POD" "$HOST" "$PORT" > "$STATE"
  echo "back up at root@$HOST:$PORT"

  echo "waiting for sshd..."
  for _ in $(seq 1 40); do pod_ssh true 2>/dev/null && break; sleep 10; done
  pod_ssh true 2>/dev/null || die "ssh never came up"

  # Re-upload the scripts so a fixed harness reaches an existing pod, then
  # relaunch. ingest9m.sh detects the populated data roots and resumes.
  scp "${SSH_OPTS[@]}" -P "$PORT" "$HERE/ingest9m.sh" "$HERE/monitor9m.sh" \
    "$HERE/idlewatch.sh" "$HERE/export9m.sh" "root@$HOST:/root/" >/dev/null || die scp
  local token_file="$HOME/.cache/huggingface/token"
  echo "relaunching the ingest (it resumes where it stopped)"
  pod_ssh "chmod +x /root/ingest9m.sh /root/idlewatch.sh; rm -f /root/RUN_DONE /root/FAILED_AT; \
    HF_TOKEN=$(cat "$token_file") ${HF_REPO:+HF_REPO=$HF_REPO} \
    WORKERS=$GPUS setsid bash /root/ingest9m.sh </dev/null >/dev/null 2>&1 & echo resumed" \
    || die "relaunch failed"
  pod_ssh "setsid bash /root/idlewatch.sh </dev/null >/dev/null 2>&1 & echo armed" || true
  echo "attaching the monitor; ctrl-b d detaches"
  sleep 3
  cmd_attach
}

cmd_watchdog() {
  read_state
  pod_ssh 'cat /root/idlewatch.log 2>/dev/null | tail -15' 2>/dev/null \
    || echo "pod is not reachable: it has most likely powered itself off (./pod9m.sh resume)"
}

cmd_down() {
  read_state
  echo "deleting pod $POD"
  runpodctl pod delete "$POD" >/dev/null 2>&1
  rm -f "$STATE"
  sleep 3
  runpodctl pod list 2>/dev/null | head -3
  echo "done. state cleared."
}

case "${1:-}" in
  up) cmd_up ;;
  attach) cmd_attach ;;
  resume) cmd_resume ;;
  watchdog) cmd_watchdog ;;
  status) cmd_status ;;
  log) cmd_log ;;
  fetch) cmd_fetch ;;
  down) cmd_down ;;
  *) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
