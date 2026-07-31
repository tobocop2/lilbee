#!/usr/bin/env bash
# Ingest 8,841,823 MS MARCO passages across every GPU on one host, with the
# telemetry a benchmark writeup needs, and resume if killed.
#
# Two of eight workers are instrumented; the rest run untouched, so headline
# throughput comes from clean workers and each instrumented worker has an
# uninstrumented twin doing identical work.
#
#   all  LILBEE_INGEST_TRACE  per-file extraction timing, every worker
#   w1   py-spy --gil         GIL-held sample fraction
#   w2   py-spy --idle        flame graph
#
# The trace covers every worker so the extraction distribution spans the whole
# corpus. It costs one scoped log call per file (the root logger is untouched),
# so at ~50 files/s per worker it is not a measurable tax.
#
# Worker isolation is environment only, no lilbee source change:
#   CUDA_VISIBLE_DEVICES        the card this worker owns
#   LILBEE_DATA                 private data root
#   LILBEE_ENGINE_DIR           private engine slot. Without it every worker
#                               adopts worker 0's fleet and the rest idle.
#   LILBEE_ANN_INDEX_THRESHOLD  0: the merge rebuilds corpus-wide anyway
#
# RESUME: re-running skips files already in a worker's store by (size, mtime)
# without rehashing. Bucket assignment is deterministic, so each worker resumes
# its own slice. Nothing here deletes a data root.
set -uo pipefail
exec >> /root/ingest.log 2>&1

: "${TAG:=v0.6.90b420.dev728}"
: "${EMBED_MODEL:=Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf}"
: "${CORPUS_URL:=https://huggingface.co/datasets/beeberg/msmarco-ingest-checkpoint/resolve/main/msmarco-passage-full.tar.gz}"
: "${EXTRACT_GLOB:=}"
: "${WORKERS:=}"
: "${EXPECTED:=}"
: "${TRACE_ALL:=1}"
: "${GIL_WORKER:=1}"
: "${FLAME_WORKER:=2}"
: "${PROFILE_RATE:=100}"
: "${MERGE:=1}"
: "${EXPORT:=1}"

die() { echo "FATAL: $*"; date -u +%s > /root/FAILED_AT; touch /root/RUN_DONE; exit 1; }
log() { printf '[ingest %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

CARDS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
CORES=$(nproc)
[ -n "$WORKERS" ] || WORKERS="$CARDS"
mkdir -p /root/status /root/prof

RESUMING=0
[ -d "/root/w0/data/lancedb" ] && RESUMING=1
log "=== $([ "$RESUMING" = 1 ] && echo RESUME || echo START) cards=$CARDS cores=$CORES workers=$WORKERS ==="

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -x /root/venv/bin/lilbee ] || uv venv --clear --seed --python 3.12 /root/venv || die venv
VPY=/root/venv/bin/python; LB=/root/venv/bin/lilbee
echo "$VPY" > /root/status/vpy

if [ ! -x /root/venv/bin/py-spy ] || [ ! -x "$LB" ]; then
  WHL=$(ls /root/payload/lilbee-*.whl 2>/dev/null | head -1)
  [ -n "$WHL" ] || die "no lilbee wheel in /root/payload"
  uv pip install -q --python "$VPY" "$WHL" py-spy || die "lilbee install"
  W="lilbee_engine-${TAG#v}-1.cu124-py3-none-manylinux_2_17_x86_64.whl"
  curl -fsSL --retry 5 -o "/tmp/$W" \
    "https://github.com/tobocop2/lilbee/releases/download/${TAG}/${W}" || die "engine wheel"
  uv pip install -q --python "$VPY" "/tmp/$W" huggingface_hub || die "engine install"
fi
"$VPY" - <<'PY' || die "import check"
import xberg
from lilbee.providers.fleet import swap_manager as sm
print(f"[ingest] xberg {getattr(xberg, '__version__', '?')} port_block={getattr(sm, '_PORT_BLOCK', 'ABSENT')}")
PY

# 8.8M small files: untar is minutes. Skipped entirely on resume.
if [ ! -d /root/corpus/documents ]; then
  [ -n "${HF_TOKEN:-}" ] || die "no HF_TOKEN"
  log "downloading corpus"
  mkdir -p /root/corpus
  curl -fsSL --retry 5 -H "Authorization: Bearer $HF_TOKEN" "$CORPUS_URL" -o /tmp/c.tgz || die tarball
  log "unpacking $(du -h /tmp/c.tgz | cut -f1)"
  tar xzf /tmp/c.tgz -C /root/corpus ${EXTRACT_GLOB:+--wildcards "$EXTRACT_GLOB"} || die untar
  rm -f /tmp/c.tgz
fi
[ -d /root/corpus/documents ] || die "no documents/ in tarball"

[ -s /root/status/buckets ] || \
  find /root/corpus/documents -mindepth 1 -maxdepth 1 -type d | sort > /root/status/buckets
NBUCK=$(wc -l < /root/status/buckets)
[ "$NBUCK" -gt 0 ] || die "corpus is not bucketed; the 1000-per-dir layout is load-bearing"
PER=$(find "$(head -1 /root/status/buckets)" -name '*.txt' -type f | wc -l)
[ -n "$EXPECTED" ] || EXPECTED=$(( NBUCK * PER ))
log "corpus: $NBUCK buckets of ~$PER = ~$EXPECTED passages"

write_cfg() {
  { echo "embedding_model = \"$EMBED_MODEL\""; echo "embedding_dim = 4096"
    echo "enable_ocr = false"; echo "embed_batch_sequences = 64"; } > "$1/config.toml"
}

if [ "$RESUMING" = 0 ]; then
  mkdir -p /root/pull; write_cfg /root/pull
  log "pulling the embedder once (models dir is shared, data roots are not)"
  LILBEE_DATA=/root/pull "$LB" model pull "$EMBED_MODEL" 2>&1 | tail -1 || die "model pull"
  # Whole buckets, original names, hard links: a worker's source keys match the
  # keys a single-host ingest produces, which keeps shards comparable to one.
  log "dealing $NBUCK buckets across $WORKERS workers"
  for i in $(seq 0 $((WORKERS-1))); do
    D="/root/w$i"; mkdir -p "$D/documents"; write_cfg "$D"
    awk -v i="$i" -v n="$WORKERS" 'NR % n == i' /root/status/buckets \
      | xargs -r -P 16 -I{} sh -c 'cp -al "$1" "$2/documents/$(basename "$1")"' _ {} "$D"
  done
fi

printf 'workers=%s\ncards=%s\ncores=%s\nexpected=%s\nembed_model=%s\nresumed=%s\n' \
  "$WORKERS" "$CARDS" "$CORES" "$EXPECTED" "$EMBED_MODEL" "$RESUMING" > /root/status/run.env

cat > /root/prof/count_rows.py <<'PYEOF'
import pathlib
import time

import lancedb

env = dict(
    line.split("=", 1)
    for line in pathlib.Path("/root/status/run.env").read_text().splitlines()
    if "=" in line
)
total = 0
for i in range(int(env.get("workers", 8))):
    try:
        db = lancedb.connect(f"/root/w{i}/data/lancedb")
        listed = db.list_tables()
        if "_sources" in list(getattr(listed, "tables", listed)):
            total += db.open_table("_sources").count_rows()
    except Exception:
        pass
print(f"{int(time.time())},{total}", flush=True)
PYEOF

worker_pids() { pgrep -f "[l]ilbee sync" | tr '\n' ',' | sed 's/,$//'; }

start_samplers() {
  ( while :; do
      nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | paste -sd, - \
        | sed "s/^/$(date -u +%s),/"
      sleep 2
    done >> /root/prof/gpu.csv ) &
  echo $! > /root/prof/.gpu.pid
  ( while :; do
      "$VPY" /root/prof/count_rows.py >> /root/prof/rows.csv 2>/dev/null
      sleep 20
    done ) &
  echo $! > /root/prof/.rows.pid
  ( while :; do
      pids=$(worker_pids)
      printf '%s,%s,%s,%s\n' "$(date -u +%s)" "$(awk '{print $1}' /proc/loadavg)" \
        "$(ps -o nlwp= -p "$pids" 2>/dev/null | awk '{s+=$1} END {print s+0}')" \
        "$(ps -o rss= -p "$pids" 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}')"
      sleep 10
    done >> /root/prof/host.csv ) &
  echo $! > /root/prof/.host.pid
}

stop_samplers() {
  for f in /root/prof/.gpu.pid /root/prof/.rows.pid /root/prof/.host.pid; do
    [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null
    rm -f "$f"
  done
}

# py-spy launching its target defeats yama ptrace_scope=1, which refuses attach
# to an unrelated pid. Proven here on a throwaway process rather than discovered
# when two workers die at launch and take the run with them.
PROFILE_OK=1
if ! /root/venv/bin/py-spy record --nonblocking --rate 10 --format raw \
     --output /tmp/spy_probe.folded -- "$VPY" -c "import time; time.sleep(2)" >/dev/null 2>&1; then
  PROFILE_OK=0
  log "py-spy cannot profile here; workers $GIL_WORKER and $FLAME_WORKER run uninstrumented"
fi
rm -f /tmp/spy_probe.folded

log "launching $WORKERS workers (trace=all gil=w$GIL_WORKER flame=w$FLAME_WORKER profile_ok=$PROFILE_OK)"
[ -f /root/status/started_at ] || date -u +%s > /root/status/started_at
date -u +%s > /root/status/attempt_started_at
start_samplers

pids=""
for i in $(seq 0 $((WORKERS-1))); do
  # py-spy must LAUNCH the worker: yama ptrace_scope=1 permits tracing a
  # descendant but refuses attach to an unrelated pid.
  spy=()
  [ "$PROFILE_OK" = "1" ] && case "$i" in
    "$GIL_WORKER")   spy=(/root/venv/bin/py-spy record --gil --nonblocking --rate "$PROFILE_RATE"
                          --format raw --output "/root/prof/w${i}.gil.folded" --) ;;
    "$FLAME_WORKER") spy=(/root/venv/bin/py-spy record --idle --nonblocking --rate "$PROFILE_RATE"
                          --format raw --output "/root/prof/w${i}.wall.folded" --) ;;
  esac
  trace_env=()
  [ "$TRACE_ALL" = "1" ] && \
    trace_env=(LILBEE_INGEST_TRACE=1 "LILBEE_INGEST_TRACE_FILE=/root/prof/w${i}.trace.log")
  ( env CUDA_VISIBLE_DEVICES="$i" LILBEE_DATA="/root/w$i" LILBEE_ENGINE_DIR="/root/w$i/engine" \
      LILBEE_ANN_INDEX_THRESHOLD=0 "${trace_env[@]}" \
      "${spy[@]}" "$LB" sync > "/root/w$i/sync.log" 2>&1
    echo "$?" > "/root/w$i/rc" ) &
  pids="$pids $!"
done
wait $pids
stop_samplers
SECS=$(( $(date -u +%s) - $(cat /root/status/attempt_started_at) ))

LANDED=0
for i in $(seq 0 $((WORKERS-1))); do
  R=$("$VPY" -c "
import lancedb
try:
    db = lancedb.connect('/root/w$i/data/lancedb')
    t = db.list_tables()
    print(db.open_table('_sources').count_rows() if '_sources' in list(getattr(t, 'tables', t)) else 0)
except Exception: print(0)" 2>/dev/null || echo 0)
  log "  worker $i: rows=$R rc=$(cat "/root/w$i/rc" 2>/dev/null)"
  LANDED=$((LANDED + R))
done
echo "INGEST workers=$WORKERS landed=$LANDED expected=$EXPECTED secs=$SECS docs_per_s=$("$VPY" -c "print(f'{$LANDED/max($SECS,1):.1f}')")"

render_profiles() {
  local fg=/root/prof/flamegraph.pl
  [ -f "$fg" ] || curl -fsSL --retry 3 -o "$fg" \
    https://raw.githubusercontent.com/brendangregg/FlameGraph/master/flamegraph.pl 2>/dev/null
  for f in /root/prof/*.folded; do
    [ -s "$f" ] || continue
    [ -f "$fg" ] && perl "$fg" --title "$(basename "$f" .folded)" "$f" > "${f%.folded}.svg" 2>/dev/null
    log "  $(basename "$f"): $(wc -l < "$f") stacks, $(du -h "$f" | cut -f1)"
  done
  "$VPY" - <<'PYEOF'
import pathlib

PROF = pathlib.Path("/root/prof")


def sample_total(name: str) -> int:
    """Summed sample counts in a py-spy folded file (``stack count`` per line)."""
    path = PROF / name
    if not path.exists():
        return 0
    total = 0
    for line in path.read_text().splitlines():
        head, _, count = line.rpartition(" ")
        if head and count.isdigit():
            total += int(count)
    return total


gil = sample_total("w1.gil.folded")
wall = sample_total("w2.wall.folded")
if wall:
    print(f"PROFILE gil_samples={gil} wall_samples={wall} gil_fraction={gil / wall:.4f}", flush=True)
else:
    print(f"PROFILE gil_samples={gil} wall_samples=0 (no wall recording)", flush=True)
PYEOF
}
render_profiles

log "compressing extraction traces"
for t in /root/prof/w*.trace.log; do
  [ -s "$t" ] || continue
  log "  $(basename "$t"): $(wc -l < "$t") extractions, $(du -h "$t" | cut -f1)"
  gzip -f "$t" 2>/dev/null &
done
wait

# An incomplete run stops here with everything on disk; re-running resumes.
if [ "$LANDED" -lt "$EXPECTED" ]; then
  log "INCOMPLETE: $LANDED of $EXPECTED. Re-run to resume; nothing is deleted."
  touch /root/RUN_DONE
  exit 0
fi

if [ "$MERGE" = "1" ]; then
  log "--- merge: $WORKERS shards into one index ---"
  SHARDS=""
  for i in $(seq 0 $((WORKERS-1))); do
    LILBEE_DATA="/root/w$i" SHARD_INDEX="$i" SHARD_COUNT="$WORKERS" DATASET_ID=msmarco-full \
      "$VPY" /root/payload/shard_manifest.py || die "manifest w$i"
    SHARDS="$SHARDS /root/w$i"
  done
  T0=$(date -u +%s); rm -rf /root/merged
  LILBEE_DATA=/root/merged "$VPY" /root/payload/merge_shards.py $SHARDS
  MRC=$?
  echo "MERGE shards=$WORKERS rc=$MRC secs=$(( $(date -u +%s) - T0 ))"
  [ "$MRC" = "0" ] || die "merge failed rc=$MRC"
fi

if [ "$EXPORT" = "1" ] && [ -d /root/merged/data/lancedb ]; then
  log "--- export and HuggingFace sync ---"
  T0=$(date -u +%s)
  if bash /root/export9m.sh; then
    echo "EXPORT rc=0 secs=$(( $(date -u +%s) - T0 ))"
  else
    echo "EXPORT rc=1 (index intact on the pod; rerun /root/export9m.sh)"
  fi
fi

log "TOTAL $(( $(date -u +%s) - $(cat /root/status/started_at) ))s across all attempts"
log DONE
touch /root/RUN_DONE
