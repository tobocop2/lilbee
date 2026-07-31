#!/usr/bin/env bash
# Find WHERE a worker's time goes as its own table grows.
#
# Three prior hypotheses (planning-pool oversubscription, per-shard ANN builds,
# the _sources predicate delete) were each argued from source, each tested on
# hardware, and each refuted. This run stops proposing mechanisms and measures
# the process directly.
#
# One run, self-controlled: the rate curve and the stack samples come from the
# SAME workers, so an early-versus-late comparison carries no cross-run noise.
#
#   1. rate curve   row counts sampled every 20s -> docs/s over time
#   2. stack dumps  py-spy dump on worker 0 every 20s -> what it is doing
#   3. flame graph  py-spy record over the whole run, --idle so blocked time shows
#   4. gpu timeline util every 2s, to line idle stretches up against the stacks
#
# The question it answers: when throughput sags, is the worker burning CPU
# somewhere, or blocked waiting, and on what.
set -uo pipefail
exec > /root/profile.log 2>&1

: "${TAG:=v0.6.90b420.dev728}"
: "${EMBED_MODEL:=Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf}"
: "${CORPUS_URL:=https://huggingface.co/datasets/beeberg/msmarco-ingest-checkpoint/resolve/main/msmarco-passage-full.tar.gz}"
: "${EXTRACT_GLOB:=documents/00[0-7]*}"
: "${CORPUS_N:=800000}"
: "${WORKERS:=8}"
: "${SAMPLE_S:=20}"
die() { echo "FATAL: $*"; touch /root/RUN_DONE; exit 1; }
log() { printf '[profile %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

CARDS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
CORES=$(nproc)
log "cards=$CARDS cores=$CORES workers=$WORKERS corpus=$CORPUS_N sample=${SAMPLE_S}s"

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --clear --seed --python 3.12 /root/venv || die venv
VPY=/root/venv/bin/python; LB=/root/venv/bin/lilbee
mkdir -p /root/status /root/prof
echo "$VPY" > /root/status/vpy

WHL=$(ls /root/payload/lilbee-*.whl 2>/dev/null | head -1)
[ -n "$WHL" ] || die "no lilbee wheel in /root/payload"
uv pip install -q --python "$VPY" "$WHL" || die "lilbee"
uv pip install -q --python "$VPY" py-spy || die "py-spy"
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
log "corpus: $NBUCK buckets of ~$PER"

write_cfg() {
  { echo "embedding_model = \"$EMBED_MODEL\""; echo "embedding_dim = 4096"
    echo "enable_ocr = false"; echo "embed_batch_sequences = 64"; } > "$1/config.toml"
}
mkdir -p /root/pull; write_cfg /root/pull
log "pulling the embedder"
LILBEE_DATA=/root/pull "$LB" model pull "$EMBED_MODEL" 2>&1 | tail -1 || die "model pull"

log "dealing buckets"
rm -rf /root/w[0-9]*
for i in $(seq 0 $((WORKERS-1))); do
  D="/root/w$i"; mkdir -p "$D/documents"; write_cfg "$D"
  awk -v i="$i" -v n="$WORKERS" 'NR % n == i' /root/status/buckets \
    | xargs -r -P 16 -I{} sh -c 'cp -al "$1" "$2/documents/$(basename "$1")"' _ {} "$D"
done
printf 'workers=%s\ncards=%s\ncores=%s\nexpected=%s\n' \
  "$WORKERS" "$CARDS" "$CORES" "$(( NBUCK * PER ))" > /root/status/run.env

log "launching $WORKERS workers"
T0=$(date -u +%s); echo "$T0" > /root/status/started_at
pids=""
for i in $(seq 0 $((WORKERS-1))); do
  ( env CUDA_VISIBLE_DEVICES="$i" LILBEE_DATA="/root/w$i" LILBEE_ENGINE_DIR="/root/w$i/engine" \
    LILBEE_ANN_INDEX_THRESHOLD=0 \
    "$LB" sync > "/root/w$i/sync.log" 2>&1; echo "$?" > "/root/w$i/rc" ) &
  pids="$pids $!"
done

# --- samplers -------------------------------------------------------------
( while :; do
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | paste -sd, - \
      | sed "s/^/$(date -u +%s),/"
    sleep 2
  done > /root/prof/gpu.csv ) &
SAMP_GPU=$!

# Row counts per worker over time: the rate curve, from the data itself rather
# than from a log line that could be throttled by the pipeline it reports on.
( while :; do
    "$VPY" - <<'PY' >> /root/prof/rows.csv 2>/dev/null
import time
import lancedb
total = 0
for i in range(8):
    try:
        db = lancedb.connect(f"/root/w{i}/data/lancedb")
        listed = db.list_tables()
        names = list(getattr(listed, "tables", listed))
        if "_sources" in names:
            total += db.open_table("_sources").count_rows()
    except Exception:
        pass
print(f"{int(time.time())},{total}", flush=True)
PY
    sleep "$SAMPLE_S"
  done ) &
SAMP_ROWS=$!

# Wait for worker 0's python to exist, then profile it for the whole run.
for _ in $(seq 1 60); do
  W0=$(pgrep -f "[l]ilbee sync" | head -1)
  [ -n "$W0" ] && break
  sleep 5
done
log "profiling pid $W0"
# --idle so time blocked on a lock or an await is attributed, not dropped: a
# starved pipeline spends its time waiting, which a CPU-only profile cannot see.
( /root/venv/bin/py-spy record --pid "$W0" --idle --format speedscope \
    --output /root/prof/worker0.speedscope --duration 5400 >/dev/null 2>&1 ) &
SAMP_FLAME=$!

# Periodic stacks: a flame graph averages the whole run and would hide a change
# over time, which is precisely what is being looked for.
( while kill -0 "$W0" 2>/dev/null; do
    echo "=== $(date -u +%s) ==="
    /root/venv/bin/py-spy dump --pid "$W0" 2>/dev/null
    sleep "$SAMPLE_S"
  done > /root/prof/stacks.txt ) &
SAMP_STACK=$!

wait $pids
SECS=$(( $(date -u +%s) - T0 ))
kill "$SAMP_GPU" "$SAMP_ROWS" "$SAMP_STACK" 2>/dev/null
wait "$SAMP_FLAME" 2>/dev/null

LANDED=0
for i in $(seq 0 $((WORKERS-1))); do
  R=$("$VPY" -c "
import lancedb
db = lancedb.connect('/root/w$i/data/lancedb')
print(db.open_table('_sources').count_rows())" 2>/dev/null || echo 0)
  LANDED=$((LANDED + R))
done
echo "PROFILE workers=$WORKERS landed=$LANDED secs=$SECS docs_per_s=$("$VPY" -c "print(f'{$LANDED/max($SECS,1):.1f}')")"

# --- the analysis this run exists for -------------------------------------
log "=== rate over time (docs/s between samples) ==="
"$VPY" - <<'PY'
import pathlib
rows = []
for line in pathlib.Path("/root/prof/rows.csv").read_text().splitlines():
    try:
        t, n = line.split(",")
        rows.append((int(t), int(n)))
    except ValueError:
        pass
if len(rows) > 2:
    t0 = rows[0][0]
    print(f"  {'elapsed':>8} {'total rows':>12} {'docs/s':>9}")
    for (ta, na), (tb, nb) in zip(rows, rows[1:]):
        dt = tb - ta
        if dt > 0:
            print(f"  {tb - t0:>7}s {nb:>12,} {(nb - na) / dt:>9.1f}")
PY

log "=== most common stacks, first quarter vs last quarter ==="
"$VPY" - <<'PY'
import collections
import pathlib

text = pathlib.Path("/root/prof/stacks.txt").read_text()
blocks = [b for b in text.split("=== ") if b.strip()]
if len(blocks) >= 4:
    q = len(blocks) // 4
    for label, chunk in (("FIRST QUARTER", blocks[:q]), ("LAST QUARTER", blocks[-q:])):
        counts: collections.Counter = collections.Counter()
        for block in chunk:
            for line in block.splitlines():
                line = line.strip()
                # py-spy dump frames look like "  fn (file.py:123)"
                if "(" in line and ".py:" in line:
                    counts[line.split(" (")[0]] += 1
        print(f"  --- {label} ({len(chunk)} samples) ---")
        for fn, n in counts.most_common(12):
            print(f"    {n:>5}  {fn}")
PY
log DONE
touch /root/RUN_DONE
