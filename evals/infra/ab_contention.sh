#!/usr/bin/env bash
# Four-arm A/B for the ingest contention finding, all rungs on ONE box in ONE run.
#
# Prior rounds compared runs across pods and the core count silently varied
# (224-core boxes gave 242 docs/s, a 160-core box gave 282 on the same corpus),
# so cross-run comparisons in this campaign are confounded. Everything here is
# measured against everything else on the same hardware.
#
# The two levers:
#   pools  LILBEE_CPU_QUOTA + LILBEE_INGEST_WORKERS pinned to cores/workers.
#          Measured on hardware: 526 threads per worker, 4208 on a 160-core box,
#          load 254, 490 threads parked on futexes while all 8 GPUs sat at 0%.
#          An earlier A/B moved only LILBEE_INGEST_WORKERS, which sizes the
#          planning pool alone and leaves kreuzberg/xberg's pool, the admission
#          window and the ingest pool untouched, so it measured ~30% of the
#          threads and came back flat.
#   batch  cfg.batch_extraction, new with the xberg 1.0 migration and off by
#          default. Every 325-byte passage otherwise takes a separate extract
#          call; batching collapses them.
#
# Each rung reports docs/s, GPU busy fraction, peak load and peak thread count,
# so a throughput change can be attributed to contention rather than guessed at.
set -uo pipefail
exec > /root/ab.log 2>&1

: "${TAG:=v0.6.90b420.dev728}"
: "${EMBED_MODEL:=Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf}"
: "${CORPUS_URL:=https://huggingface.co/datasets/beeberg/msmarco-ingest-checkpoint/resolve/main/msmarco-passage-full.tar.gz}"
: "${EXTRACT_GLOB:=documents/00[0-3]*}"
: "${CORPUS_N:=400000}"
: "${WORKERS:=8}"
: "${BATCH_SIZE:=32}"
: "${ARMS:=baseline pools batch both}"
die() { echo "FATAL: $*"; touch /root/RUN_DONE; exit 1; }
log() { printf '[ab %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

CARDS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
CORES=$(nproc)
SHARE=$(( CORES / WORKERS ))
log "cards=$CARDS cores=$CORES workers=$WORKERS per-worker-share=$SHARE arms='$ARMS'"

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --clear --seed --python 3.12 /root/venv || die venv
VPY=/root/venv/bin/python; LB=/root/venv/bin/lilbee
mkdir -p /root/status; echo "$VPY" > /root/status/vpy

WHL=$(ls /root/payload/lilbee-*.whl 2>/dev/null | head -1)
[ -n "$WHL" ] || die "no lilbee wheel in /root/payload"
uv pip install -q --python "$VPY" "$WHL" || die lilbee
"$VPY" - <<'PY' || die "import check"
from lilbee.providers.fleet import swap_manager as sm
print(f"[ab] port block={getattr(sm, '_PORT_BLOCK', 'ABSENT')}")
import xberg
print(f"[ab] xberg {getattr(xberg, '__version__', 'present')}")
from lilbee.core.config import cfg
print(f"[ab] batch_extraction default={cfg.batch_extraction}")
PY
W="lilbee_engine-${TAG#v}-1.cu124-py3-none-manylinux_2_17_x86_64.whl"
curl -fsSL --retry 5 -o "/tmp/$W" \
  "https://github.com/tobocop2/lilbee/releases/download/${TAG}/${W}" || die "engine wheel"
uv pip install -q --python "$VPY" "/tmp/$W" huggingface_hub || die engine

log "fetching corpus"
[ -n "${HF_TOKEN:-}" ] || die "no HF_TOKEN"
mkdir -p /root/corpus
curl -fsSL --retry 5 -H "Authorization: Bearer $HF_TOKEN" "$CORPUS_URL" -o /tmp/c.tgz || die tarball
tar xzf /tmp/c.tgz -C /root/corpus ${EXTRACT_GLOB:+--wildcards "$EXTRACT_GLOB"} || die untar
find /root/corpus/documents -mindepth 1 -maxdepth 1 -type d | sort > /root/status/buckets.all
PER=$(find "$(head -1 /root/status/buckets.all)" -name '*.txt' -type f | wc -l)
head -n $(( (CORPUS_N + PER - 1) / PER )) /root/status/buckets.all > /root/status/buckets
NBUCK=$(wc -l < /root/status/buckets)
INPUT=$(( NBUCK * PER ))
log "corpus: $NBUCK buckets of ~$PER = ~$INPUT passages"

write_cfg() {
  { echo "embedding_model = \"$EMBED_MODEL\""; echo "embedding_dim = 4096"
    echo "enable_ocr = false"; echo "embed_batch_sequences = 64"; } > "$1/config.toml"
}
mkdir -p /root/pull; write_cfg /root/pull
log "pulling the embedder once"
LILBEE_DATA=/root/pull "$LB" model pull "$EMBED_MODEL" 2>&1 | tail -1 || die "model pull"

run_arm() {
  local arm="$1" i
  # Per-arm environment. Unset means lilbee's own default, which is the point of
  # the baseline arm: it must not be given any of the knobs under test.
  local q="" w="" b="" bs=""
  case "$arm" in
    baseline) ;;
    pools)    q="$SHARE"; w="$SHARE" ;;
    batch)    b="true"; bs="$BATCH_SIZE" ;;
    both)     q="$SHARE"; w="$SHARE"; b="true"; bs="$BATCH_SIZE" ;;
    *) log "unknown arm $arm"; return ;;
  esac
  log "--- arm '$arm': cpu_quota=${q:-default} ingest_workers=${w:-default} batch=${b:-off} ---"

  rm -rf /root/w[0-9]* /tmp/gpu.samples /tmp/load.samples
  for i in $(seq 0 $((WORKERS-1))); do
    local D="/root/w$i"; mkdir -p "$D/documents"; write_cfg "$D"
    awk -v i="$i" -v n="$WORKERS" 'NR % n == i' /root/status/buckets \
      | xargs -r -P 16 -I{} sh -c 'cp -al "$1" "$2/documents/$(basename "$1")"' _ {} "$D"
  done

  ( for _ in $(seq 1 3000); do
      nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | paste -sd, -
      sleep 2
    done > /tmp/gpu.samples ) &
  local sampler=$!
  # Load and thread count are what make a throughput delta attributable: if the
  # fast arm also shows lower load and fewer threads, contention is the cause.
  ( for _ in $(seq 1 600); do
      printf '%s,%s\n' "$(awk '{print $1}' /proc/loadavg)" \
        "$(ps -o nlwp= -p $(pgrep -f '[l]ilbee sync' | tr '\n' ',' | sed 's/,$//') 2>/dev/null | awk '{s+=$1} END {print s+0}')"
      sleep 10
    done > /tmp/load.samples ) &
  local loadsamp=$!

  local t0 pids=""
  t0=$(date -u +%s)
  for i in $(seq 0 $((WORKERS-1))); do
    ( env CUDA_VISIBLE_DEVICES="$i" LILBEE_DATA="/root/w$i" LILBEE_ENGINE_DIR="/root/w$i/engine" \
      LILBEE_ANN_INDEX_THRESHOLD=0 \
      ${q:+LILBEE_CPU_QUOTA="$q"} ${w:+LILBEE_INGEST_WORKERS="$w"} \
      ${b:+LILBEE_BATCH_EXTRACTION="$b"} ${bs:+LILBEE_BATCH_EXTRACTION_SIZE="$bs"} \
      "$LB" sync > "/root/w$i/sync.log" 2>&1; echo "$?" > "/root/w$i/rc" ) &
    pids="$pids $!"
  done
  wait $pids
  local secs=$(( $(date -u +%s) - t0 ))
  kill "$sampler" "$loadsamp" 2>/dev/null

  local landed=0 rows
  for i in $(seq 0 $((WORKERS-1))); do
    rows=$("$VPY" -c "
import lancedb
try:
    db = lancedb.connect('/root/w$i/data/lancedb')
    t = db.list_tables()
    print(db.open_table('_sources').count_rows() if '_sources' in list(getattr(t, 'tables', t)) else 0)
except Exception: print(0)" 2>/dev/null || echo 0)
    landed=$((landed + rows))
  done

  "$VPY" - "$arm" "$landed" "$secs" <<'PY'
import sys
arm, landed, secs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
def read(path, cast=float):
    try:
        return [l.strip() for l in open(path) if l.strip()]
    except OSError:
        return []
gpu = [[int(x) for x in l.split(",")] for l in read("/tmp/gpu.samples") if l[0].isdigit()]
busy = [g for g in gpu if sum(g) / len(g) > 10]
frac = len(busy) / len(gpu) if gpu else 0
util = sum(sum(g) / len(g) for g in busy) / len(busy) if busy else 0
loads, threads = [], []
for line in read("/tmp/load.samples"):
    try:
        a, b = line.split(",")
        loads.append(float(a)); threads.append(int(b))
    except ValueError:
        pass
print(
    f"AB arm={arm} landed={landed} secs={secs} "
    f"docs_per_s={landed / max(secs, 1):.1f} busy_frac={frac:.2f} util_busy={util:.0f} "
    f"peak_load={max(loads) if loads else 0:.0f} peak_threads={max(threads) if threads else 0}",
    flush=True,
)
PY
}

for arm in $ARMS; do run_arm "$arm"; done
log "=== SUMMARY ==="
grep -aE "^AB arm=" /root/ab.log
log DONE
touch /root/RUN_DONE
