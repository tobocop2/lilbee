#!/usr/bin/env bash
# N lilbee ingest workers, one per GPU, on one host. Optionally sweeps N.
#
# Worker isolation is environment only; no lilbee source change:
#   CUDA_VISIBLE_DEVICES  the card this worker owns
#   LILBEE_DATA           private data root (config, documents, lancedb)
#   LILBEE_ENGINE_DIR     private engine slot. Without it every worker scans the
#                         machine-wide slot machine_engine_dir() returns, finds
#                         worker 0's live fleet and adopts it, which leaves one
#                         card at 95% and the rest at 0%.
#   LILBEE_INGEST_WORKERS optional planning-pool share. Unset, each worker sizes
#                         its pool to the whole box. Unset is what the 98.5
#                         docs/s 2xH100 baseline used, so it stays the default
#                         and PLAN_POOL is an explicit input.
#
# The corpus is split by whole bucket directories, each keeping its original
# name, so a worker's source keys are the keys a single-host ingest of the same
# corpus produces and the shards can be compared against one. Bucketing is
# load-bearing, not cosmetic: 10k files in ONE directory measured 1334s/15.0
# docs/s against 401s/49.9 bucketed 1000/dir, because a starved discovery scan
# leaves the GPU idle on a pull-driven plan stream.
#
# Ports need no stagger since each lilbee takes its own block of the
# sub-ephemeral window. STAGGER_S reproduces the old workaround for an A/B.
set -uo pipefail
exec > /root/pergpu.log 2>&1

: "${TAG:=v0.6.90b420.dev728}"
: "${EMBED_MODEL:=Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf}"
: "${CORPUS_URL:=https://huggingface.co/datasets/beeberg/msmarco-ingest-checkpoint/resolve/main/msmarco-passage-80k.tar.gz}"
: "${CORPUS_N:=20000}"   # passages to use; 0 = the whole tarball
: "${BUCKET:=1000}"      # files per bucket when the tarball is flat
: "${EXTRACT_GLOB:=}"    # tar pattern to limit which buckets are unpacked
: "${WORKERS:=}"         # rungs to run, e.g. "1 2 4 8"; default = card count
: "${STAGGER_S:=0}"
: "${PLAN_POOL:=}"
: "${REFERENCE:=0}"      # 1: two single-card whole-corpus ingests first, as the
                         # correctness reference and its own drift floor
: "${MERGE:=0}"          # 1: merge the last rung's shards and compare them to
                         # the reference (needs REFERENCE=1 and the payload tools)
: "${POOL_RUNG:=0}"      # 1: repeat the widest rung with the planning pool divided
die() { echo "FATAL: $*"; touch /root/RUN_DONE; exit 1; }
log() { printf '[pergpu %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

CARDS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
CORES=$(nproc)
[ -n "$WORKERS" ] || WORKERS="$CARDS"
log "cards=$CARDS cores=$CORES rungs='$WORKERS' stagger=${STAGGER_S}s plan_pool=${PLAN_POOL:-lilbee-default}"

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --clear --seed --python 3.12 /root/venv || die venv
VPY=/root/venv/bin/python; LB=/root/venv/bin/lilbee

# A wheel in /root/payload is a local build under test; main is the fallback.
WHL=$(ls /root/payload/lilbee-*.whl 2>/dev/null | head -1)
if [ -n "$WHL" ]; then
  log "installing lilbee from payload wheel $(basename "$WHL")"
  uv pip install -q --python "$VPY" "$WHL" || die "lilbee wheel"
else
  log "installing lilbee@main"
  uv pip install -q --python "$VPY" "git+https://github.com/tobocop2/lilbee@main" || die lilbee
fi
"$VPY" - <<'PY' || die "port block check"
from lilbee.providers.fleet import swap_manager as sm
blocked = hasattr(sm, "_PORT_BLOCK")
print(f"[pergpu] port blocks present: {blocked}")
PY

W="lilbee_engine-${TAG#v}-1.cu124-py3-none-manylinux_2_17_x86_64.whl"
curl -fsSL --retry 5 -o "/tmp/$W" \
  "https://github.com/tobocop2/lilbee/releases/download/${TAG}/${W}" || die wheel
uv pip install -q --python "$VPY" "/tmp/$W" huggingface_hub || die engine

log "fetching corpus"
mkdir -p /root/corpus
[ -n "${HF_TOKEN:-}" ] || die "no HF_TOKEN"
curl -fsSL --retry 5 -H "Authorization: Bearer $HF_TOKEN" "$CORPUS_URL" -o /tmp/c.tgz || die tarball
# EXTRACT_GLOB pulls only the buckets a run needs: the full tarball holds 8.8M
# files and writing all of them to use a tenth is minutes of inodes for nothing.
tar xzf /tmp/c.tgz -C /root/corpus ${EXTRACT_GLOB:+--wildcards "$EXTRACT_GLOB"} || die untar
SRC=/root/corpus/documents
[ -d "$SRC" ] || die "no documents/ in tarball"

# Buckets are the unit of work. A tarball that is already bucketed keeps its
# names; a flat one is bucketed once here so every rung splits the same tree.
find "$SRC" -mindepth 1 -maxdepth 1 -type d | sort > /tmp/buckets.all
if [ ! -s /tmp/buckets.all ]; then
  log "tarball is flat: bucketing $BUCKET files per directory"
  NEW=/root/corpus/bucketed
  # Trim to CORPUS_N before bucketing: bucketing every file to then use a
  # twentieth of them is minutes of link calls for nothing.
  find "$SRC" -maxdepth 1 -name '*.txt' -type f | sort \
    | { [ "$CORPUS_N" -gt 0 ] && head -n "$CORPUS_N" || cat; } \
    | awk -v d="$NEW" -v b="$BUCKET" '{ printf "%s/%05d\t%s\n", d, int((NR-1)/b), $0 }' > /tmp/deal.all
  cut -f1 /tmp/deal.all | sort -u | xargs -r mkdir -p
  while IFS=$'\t' read -r bucket f; do ln "$f" "$bucket/$(basename "$f")" 2>/dev/null; done < /tmp/deal.all
  SRC="$NEW"
  find "$SRC" -mindepth 1 -maxdepth 1 -type d | sort > /tmp/buckets.all
fi
PER=$(find "$(head -1 /tmp/buckets.all)" -name '*.txt' -type f | wc -l)
if [ "$CORPUS_N" -gt 0 ]; then
  KEEP=$(( (CORPUS_N + PER - 1) / PER ))
  head -n "$KEEP" /tmp/buckets.all > /tmp/buckets
else
  cp /tmp/buckets.all /tmp/buckets
fi
NBUCK=$(wc -l < /tmp/buckets)
INPUT=$(( NBUCK * PER ))
log "corpus: $NBUCK buckets of ~$PER = ~$INPUT passages (of $(wc -l < /tmp/buckets.all) buckets on disk)"

write_cfg() {
  { echo "embedding_model = \"$EMBED_MODEL\""; echo "embedding_dim = 4096"
    echo "enable_ocr = false"; echo "embed_batch_sequences = 64"
    # No per-shard ANN index. Each worker would build an IVF_PQ over its own
    # vectors and the merge rebuilds one corpus-wide anyway (force=True), so the
    # per-shard build is thrown away. Measured: 10k rows/worker sits under the
    # 50k default and idled the GPUs 25% of the run; 100k rows/worker crosses it
    # and idled them 45%.
    echo "ann_index_threshold = ${ANN_THRESHOLD:-0}"; } > "$1/config.toml"
}

mkdir -p /root/pull; write_cfg /root/pull
log "pulling the embedder once (the models dir is shared, the data root is not)"
LILBEE_DATA=/root/pull "$LB" model pull "$EMBED_MODEL" 2>&1 | tail -2 || die "model pull"

# One rung: N workers over the same bucket set, fresh data roots, GPU sampled.
run_rung() {
  local N="$1" i
  rm -rf /root/w[0-9]* /tmp/gpu.samples
  log "--- rung N=$N: dealing $NBUCK buckets across $N workers ---"
  for i in $(seq 0 $((N-1))); do
    local D="/root/w$i"
    mkdir -p "$D/documents"; write_cfg "$D"
    awk -v i="$i" -v n="$N" 'NR % n == i' /tmp/buckets \
      | xargs -r -P 16 -I{} sh -c 'cp -al "$1" "$2/documents/$(basename "$1")"' _ {} "$D"
    log "  worker $i: $(find "$D/documents" -name '*.txt' -type f | wc -l) passages in $(find "$D/documents" -mindepth 1 -type d | wc -l) buckets"
  done

  ( for _ in $(seq 1 5400); do
      nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd, -
      sleep 2
    done > /tmp/gpu.samples ) &
  local sampler=$!

  log "launching $N workers (stagger ${STAGGER_S}s)"
  local t0 pids=""
  t0=$(date +%s)
  for i in $(seq 0 $((N-1))); do
    ( env CUDA_VISIBLE_DEVICES="$i" LILBEE_DATA="/root/w$i" LILBEE_ENGINE_DIR="/root/w$i/engine" \
      LILBEE_ANN_INDEX_THRESHOLD="${ANN_THRESHOLD:-0}" \
      ${PLAN_POOL:+LILBEE_INGEST_WORKERS="$PLAN_POOL"} \
      "$LB" sync > "/root/w$i/sync.log" 2>&1; echo "$?" > "/root/w$i/rc" ) &
    pids="$pids $!"
    [ "$STAGGER_S" -gt 0 ] && [ "$i" -lt $((N-1)) ] && sleep "$STAGGER_S"
  done
  # Wait on the workers only: a bare wait also waits for the sampler, which
  # outlives them by half an hour and would swallow the summary.
  wait $pids
  local secs=$(( $(date +%s) - t0 ))
  kill "$sampler" 2>/dev/null

  local landed=0 rows
  for i in $(seq 0 $((N-1))); do
    rows=$("$VPY" -c "
import lancedb
try:
    db = lancedb.connect('/root/w$i/data/lancedb')
    t = db.list_tables()
    print(db.open_table('_sources').count_rows() if '_sources' in list(getattr(t, 'tables', t)) else 0)
except Exception: print(0)" 2>/dev/null || echo 0)
    log "  worker $i: rows=$rows rc=$(cat "/root/w$i/rc" 2>/dev/null) $(grep -ac 'Traceback' "/root/w$i/sync.log" 2>/dev/null) tracebacks"
    landed=$((landed + rows))
  done

  local gpu
  gpu=$("$VPY" -c "
rows = [l.strip() for l in open('/tmp/gpu.samples') if l.strip()]
v = [[int(x) for x in r.split(',')] for r in rows]
if not v: print('n/a')
else:
    n = len(v[0])
    per = [sum(r[i] for r in v) / len(v) for i in range(n)]
    busy = [r for r in v if sum(r) / n > 10]
    pb = [sum(r[i] for r in busy) / len(busy) for i in range(n)] if busy else []
    print(f\"whole_run={'/'.join(f'{p:.0f}' for p in per)} embedding={'/'.join(f'{p:.0f}' for p in pb)} busy_frac={len(busy)/len(v):.2f}\")" 2>/dev/null || echo n/a)

  echo "PERGPU workers=$N pool=${PLAN_POOL:-default} input=$INPUT landed=$landed secs=$secs end_to_end=$("$VPY" -c "print(f'{${landed:-0}/max($secs,1):.1f}')") gpu=$gpu"
}

# The correctness reference: one lilbee, one card, the WHOLE corpus, which is
# what a single-host run is. Two of them run at once on separate cards because
# the second is the drift floor: embeddings are not deterministic, so a single
# host against its own re-run is the only scale a merged-vs-single number can be
# read against. They are independent processes, so running them side by side
# costs half the wall clock and changes neither's result.
run_reference() {
  local i
  log "--- reference: 2 single-card whole-corpus ingests (cards 0,1) ---"
  local t0 pids=""
  for i in 0 1; do
    local D="/root/ref$i"
    rm -rf "$D"; mkdir -p "$D/documents"; write_cfg "$D"
    xargs -r -P 16 -I{} sh -c 'cp -al "$1" "$2/documents/$(basename "$1")"' _ {} "$D" < /tmp/buckets
    log "  reference $i: $(find "$D/documents" -name '*.txt' -type f | wc -l) passages"
  done
  t0=$(date +%s)
  for i in 0 1; do
    ( env CUDA_VISIBLE_DEVICES="$i" LILBEE_DATA="/root/ref$i" LILBEE_ENGINE_DIR="/root/ref$i/engine" \
      LILBEE_ANN_INDEX_THRESHOLD="${ANN_THRESHOLD:-0}" \
      ${PLAN_POOL:+LILBEE_INGEST_WORKERS="$PLAN_POOL"} \
      "$LB" sync > "/root/ref$i/sync.log" 2>&1; echo "$?" > "/root/ref$i/rc" ) &
    pids="$pids $!"
  done
  wait $pids
  local secs=$(( $(date +%s) - t0 ))
  for i in 0 1; do
    log "  reference $i: rows=$("$VPY" -c "
import lancedb
db = lancedb.connect('/root/ref$i/data/lancedb')
print(db.open_table('_sources').count_rows())" 2>/dev/null || echo 0) rc=$(cat "/root/ref$i/rc" 2>/dev/null)"
  done
  echo "REFERENCE cards=2 each_input=$INPUT secs=$secs per_card=$("$VPY" -c "print(f'{$INPUT/max($secs,1):.1f}')")"
}

[ "$REFERENCE" = "1" ] && run_reference

LAST_N=0
for N in $WORKERS; do
  [ "$N" -le "$CARDS" ] || { log "skipping N=$N: only $CARDS cards"; continue; }
  run_rung "$N"
  LAST_N="$N"
done

# Unset, every worker sizes its planning pool to the whole box, so N workers ask
# for N times the host's cores. That is harmless at N=2 and unmeasured at N=8, so
# the widest rung repeats with the pool divided. Without this a flat rung cannot
# be told apart from planning oversubscription.
if [ "$POOL_RUNG" = "1" ] && [ "$LAST_N" -ge 2 ]; then
  PLAN_POOL=$(( CORES / LAST_N ))
  log "repeating N=$LAST_N with the planning pool divided ($PLAN_POOL per worker)"
  run_rung "$LAST_N"
fi

# The merge is folded into this run because the shards are already local here;
# colocating them anywhere else is the expensive part.
if [ "$MERGE" = "1" ] && [ "$LAST_N" -ge 2 ]; then
  log "--- merge: $LAST_N shards ---"
  SHARDS=""
  for i in $(seq 0 $((LAST_N-1))); do
    LILBEE_DATA="/root/w$i" SHARD_INDEX="$i" SHARD_COUNT="$LAST_N" DATASET_ID="msmarco-pergpu" \
      "$VPY" /root/payload/shard_manifest.py || die "manifest w$i"
    SHARDS="$SHARDS /root/w$i"
  done
  T0=$(date +%s)
  rm -rf /root/merged
  LILBEE_DATA=/root/merged "$VPY" /root/payload/merge_shards.py $SHARDS
  MRC=$?
  echo "MERGE shards=$LAST_N rc=$MRC secs=$(( $(date +%s) - T0 ))"
  if [ "$MRC" = "0" ] && [ -d /root/ref0 ]; then
    log "--- compare: merged vs single-host, floor = the reference re-run ---"
    "$VPY" /root/payload/compare_index.py /root/ref0 /root/merged /root/ref1
  fi
fi

# The last rung's shards stay on disk for the merge/commit step to work on.
log "shards left in place: $(ls -d /root/w[0-9]* 2>/dev/null | tr '\n' ' ')"
log DONE
touch /root/RUN_DONE
