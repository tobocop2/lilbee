#!/usr/bin/env bash
# Full-corpus MS MARCO ingest through lilbee's native per-GPU fan-out (PR #644).
#
# One bare `lilbee sync`. No CUDA_VISIBLE_DEVICES, no private data roots, no
# private engine slots, no corpus dealing, no merge script: the branch spawns one
# worker per card, deals the corpus by predicate and folds the shards itself.
# Everything this script still does is measurement, publishing and survival.
#
# WHERE THINGS LIVE, and why the split matters. The index sits on a RunPod
# network volume mounted at /workspace, so losing the pod stops meaning losing
# the run: a fresh pod in the same datacenter re-attaches the volume and `sync`
# resumes. The corpus sits on the container disk, because it is 8.8M files of
# ~325 bytes each and a walk of that over a network filesystem is the slowest
# thing here; it is also a 1.3GB tarball away from being rebuilt. documents_dir
# is a writable config field, so the two can live on different filesystems.
#
# EXPORT IS NEVER GATED ON A COUNT CHECK. The previous full run completed
# perfectly and was destroyed by its own guard: EXPECTED was computed as
# buckets x files-per-bucket, which over-counts by 177 because the last bucket
# holds 823 files, so the guard fired, the merge and export were skipped, and the
# watchdog stopped the pod. Here the count is measured rather than extrapolated
# AND a mismatch only annotates the summary. A counting bug can cost accuracy in
# a report; it can no longer cost the artifacts.
#
# This script ends at MERGE_DONE. Publishing runs from publish9m.sh on a cheap
# CPU pod attached to the same volume, because a 150GB upload does not need eight
# H100s idling behind it at $21.52/hr.
set -uo pipefail

: "${VOL:=/workspace}"                 # network volume: survives the pod
: "${CORPUS_DIR:=/root/corpus}"        # container disk: fast, rebuildable
: "${BRANCH:=feat/native-multi-gpu-ingest}"
: "${TAG:=v0.6.90b420.dev728}"
: "${EMBED_MODEL:=Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf}"
: "${EMBED_DIM:=4096}"
: "${CORPUS_URL:=https://huggingface.co/datasets/beeberg/msmarco-ingest-checkpoint/resolve/main/msmarco-passage-full.tar.gz}"
: "${EXTRACT_GLOB:=}"                  # e.g. 'documents/000*' for a trial
: "${TRACE:=1}"                        # per-file extraction trace
: "${PROFILE:=1}"                      # py-spy on two workers
: "${PROFILE_RATE:=25}"
: "${ANN:=1}"                          # 0 skips the corpus-wide ANN/FTS build

KB="$VOL/kb"; DATA="$KB/.lilbee"
PROF="$VOL/prof"; STATUS="$VOL/status"
mkdir -p "$VOL" "$PROF" "$STATUS" "$DATA" "$VOL/models" 2>/dev/null

exec >> "$VOL/ingest.log" 2>&1
log()   { printf '[ingest %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
phase() { echo "$*" > "$STATUS/phase"; log "=== $* ==="; }
die()   { log "FATAL: $*"; date -u +%s > "$VOL/FAILED_AT"; echo "$*" > "$STATUS/fatal"; exit 1; }

[ -d "$VOL" ] || die "no network volume at $VOL"
# A volume that is really the container disk defeats the whole point, and the
# symptom (everything works, then the pod dies) arrives six hours too late.
findmnt -no SOURCE --target "$VOL" > "$STATUS/volume_source" 2>/dev/null
log "volume $VOL backed by $(cat "$STATUS/volume_source" 2>/dev/null || echo unknown)"

CARDS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
CORES=$(nproc)
log "cards=$CARDS cores=$CORES branch=$BRANCH volume=$VOL"
[ "$CARDS" -ge 1 ] || die "no GPUs visible"

# --- install ------------------------------------------------------------------
phase "installing lilbee"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
VPY=/root/venv/bin/python; LB=/root/venv/bin/lilbee
if [ ! -x "$VPY" ]; then
  uv venv --clear --seed --python 3.12 /root/venv || die "venv"
fi
if ! "$VPY" -c "import lilbee" 2>/dev/null; then
  WHL=$(ls /root/payload/lilbee-*.whl 2>/dev/null | head -1)
  if [ -n "$WHL" ]; then
    uv pip install -q --python "$VPY" --prerelease=allow "$WHL" || die "wheel install"
  else
    uv pip install -q --python "$VPY" --prerelease=allow \
      "git+https://github.com/tobocop2/lilbee@${BRANCH}" || die "branch install"
  fi
  W="lilbee_engine-${TAG#v}-1.cu124-py3-none-manylinux_2_17_x86_64.whl"
  curl -fsSL --retry 5 -o "/tmp/$W" \
    "https://github.com/tobocop2/lilbee/releases/download/${TAG}/${W}" || die "engine download"
  uv pip install -q --python "$VPY" "/tmp/$W" huggingface_hub || die "engine install"
fi
uv pip install -q --python "$VPY" py-spy >/dev/null 2>&1
echo "$VPY" > "$STATUS/vpy"

# The fan-out is the whole point of this run, so its presence is checked rather
# than assumed: a wheel that predates the branch would silently ingest single-process.
"$VPY" - <<'PY' || die "the installed lilbee has no native fan-out"
import importlib.metadata as md
import pathlib

import lilbee_engine
from lilbee.core.config.model import Config
from lilbee.data.ingest.fanout import plan_fanout, resolve_process_count  # noqa: F401

# Versions come from package metadata: lilbee's __init__ is lazy and raises
# AttributeError for __version__, so reading the dunder fails a check that the
# imports above have already passed.
print(f"[ingest] lilbee {md.version('lilbee')} xberg {md.version('xberg')}")
print(f"[ingest] ingest_processes default={Config.model_fields['ingest_processes'].default}")
engine = pathlib.Path(lilbee_engine.get_llama_server_path())
print(f"[ingest] engine {engine} {engine.is_file()}")
raise SystemExit(0 if engine.is_file() else 1)
PY

# --- corpus -------------------------------------------------------------------
# On the container disk, and skipped when a restarted pod already holds it.
phase "staging the corpus"
# Located rather than assumed: the tarballs in this repo are not all rooted the
# same way, and a wrong guess here produces an empty ingest rather than an error.
find_docs() { find "$CORPUS_DIR" -maxdepth 3 -type d -name documents 2>/dev/null | head -1; }
DOCS=$(find_docs)
if [ -z "$DOCS" ] || [ -z "$(ls -A "$DOCS" 2>/dev/null)" ]; then
  [ -n "${HF_TOKEN:-}" ] || die "no HF_TOKEN"
  mkdir -p "$CORPUS_DIR"
  # The tarball is cached on the VOLUME, the unpacked tree is not. A replacement
  # pod has to re-unpack (the 8.8M-file tree belongs on the container disk) but
  # never has to re-download, and 1.3GB on the volume is free next to the index.
  TARBALL="$VOL/corpus.tgz"
  if [ ! -s "$TARBALL" ]; then
    curl -fsSL --retry 8 --retry-delay 5 -H "Authorization: Bearer $HF_TOKEN" \
      "$CORPUS_URL" -o "$TARBALL.part" || die "corpus download"
    mv "$TARBALL.part" "$TARBALL"
  fi
  # shellcheck disable=SC2086  # EXTRACT_GLOB is deliberately word-split
  tar xzf "$TARBALL" -C "$CORPUS_DIR" \
    ${EXTRACT_GLOB:+--wildcards $EXTRACT_GLOB} || die "untar"
  DOCS=$(find_docs)
fi
[ -n "$DOCS" ] && [ -d "$DOCS" ] || die "no documents/ under $CORPUS_DIR"
echo "$DOCS" > "$STATUS/documents_dir"

# COUNT the corpus; never extrapolate from one bucket. The last bucket holds a
# remainder (8,841,823 is 8841 full buckets plus 823), which is exactly how the
# previous run's guard fired on a perfect ingest. Cached because the walk is
# minutes over 8.8M files and a resume must not repeat it.
phase "counting the corpus"
if [ ! -s "$STATUS/expected" ]; then
  find "$DOCS" -type f -name '*.txt' -printf '.' | wc -c > "$STATUS/expected"
fi
EXPECTED=$(cat "$STATUS/expected")
[ "${EXPECTED:-0}" -gt 0 ] || die "counted zero files under $DOCS"
log "corpus: $EXPECTED passages in $(find "$DOCS" -mindepth 1 -maxdepth 1 -type d | wc -l) buckets"

# --- configuration ------------------------------------------------------------
# Every knob this run sets, and why, in one place. Nothing else is exported into
# lilbee's environment: the branch derives the per-worker setup itself.
phase "writing the run configuration"
{
  echo "embedding_model = \"$EMBED_MODEL\""
  echo "embedding_dim = $EMBED_DIM"
  echo "documents_dir = \"$DOCS\""
  echo "enable_ocr = false"
  echo "embed_batch_sequences = 64"
  # 0 = one worker per card, which is what this run measures.
  echo "ingest_processes = 0"
  # Leiden clustering over 8.8M chunks is a post-ingest pass this run has no use
  # for and no measurement of; on by default, so it is turned off explicitly.
  echo "concept_graph = false"
  echo "wiki = false"
  echo "entity_extraction = false"
  [ "$ANN" = "1" ] || echo "ann_index_threshold = 0"
} > "$DATA/config.toml"
cp "$DATA/config.toml" "$STATUS/config.toml"

# The 8GB embedder on the volume, so a replacement pod does not re-download it.
export LILBEE_MODELS_DIR="$VOL/models"
phase "pulling the embedder"
# tr before tail: the download bar redraws with \r, so a plain tail keeps one
# "line" holding hundreds of redraws and buries the log in it.
"$LB" model pull "$EMBED_MODEL" 2>&1 | tr '\r' '\n' | grep -a . | tail -2 || die "model pull"

# --- samplers -----------------------------------------------------------------
# Written to the volume so a lost pod loses no measurement, and started before
# the ingest so the ramp is captured rather than inferred.
phase "starting samplers"
SAMPLERS=()
( while :; do
    printf '%s,%s\n' "$(date -u +%s)" \
      "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | paste -sd, -)"
    sleep 2
  done ) >> "$PROF/gpu.csv" 2>/dev/null & SAMPLERS+=($!)
( while :; do
    printf '%s,%s,%s,%s\n' "$(date -u +%s)" "$(awk '{print $1}' /proc/loadavg)" \
      "$(awk '/^Threads:/{s+=$2} END{print s+0}' /proc/[0-9]*/status 2>/dev/null)" \
      "$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)"
    sleep 10
  done ) >> "$PROF/host.csv" 2>/dev/null & SAMPLERS+=($!)
( while :; do
    printf '%s,%s,%s\n' "$(date -u +%s)" \
      "$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | paste -sd+ - | bc 2>/dev/null)" \
      "$(df -B1 --output=used "$VOL" | tail -1)"
    sleep 30
  done ) >> "$PROF/sys.csv" 2>/dev/null & SAMPLERS+=($!)
# Its stderr goes to its own file: lancedb emits deprecation warnings per
# connect, and 1000 ticks of them buries the run log they would otherwise share.
"$VPY" /root/rows_sampler.py "$DATA" "$PROF/rows.csv" 20 2>"$PROF/rows_sampler.err" &
SAMPLERS+=($!)
echo "${SAMPLERS[*]}" > "$STATUS/samplers"

# --- ingest -------------------------------------------------------------------
phase "ingest: lilbee sync across $CARDS cards"
{
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cards=$CARDS cores=$CORES expected=$EXPECTED"
  echo "model=$EMBED_MODEL dim=$EMBED_DIM branch=$BRANCH ann=$ANN"
  echo "volume=$(cat "$STATUS/volume_source" 2>/dev/null)"
} > "$STATUS/run.env"

trace_env=()
[ "$TRACE" = "1" ] && trace_env=(LILBEE_INGEST_TRACE=1
                                 "LILBEE_INGEST_TRACE_FILE=$PROF/extract.trace.log")
T0=$(date -u +%s); echo "$T0" > "$STATUS/ingest_started"
mkdir -p "$KB"
# Run under a pty. lilbee's aggregate progress bar is a rich Live display, and
# rich renders nothing at all into a redirected file, so a plain `> sync.out`
# captures the startup warning and then six hours of silence.
#
# script's typescript goes to /dev/null and its stdout is filtered instead,
# because that bar redraws about ten times a second: recorded verbatim it is
# ~2KB/s, which is tens of gigabytes over a full run. The filter keeps a
# one-line live file for the dashboard to tail, and appends everything that is
# NOT the bar to a full log, so warnings and errors are still all kept.
( cd "$KB" && TERM=xterm-256color env "${trace_env[@]}" \
    script -qfec "$LB sync" /dev/null 2>&1 \
  | tr '\r' '\n' \
  | awk -v live="$VOL/sync.out" -v full="$VOL/sync.full.log" '
      /[^[:space:]]/ { line = $0 }
      /[^[:space:]]/ && !/Ingesting on [0-9]+ workers/ { print > full; fflush(full) }
      { if (++n % 25 == 0) { print line > live; close(live) } }
      END { print line > live; close(live) }'
  # script -e returns the command's own exit code, and PIPESTATUS[0] is how it
  # gets out of the pipeline: without this the subshell reports awk's status and
  # a failed sync looks like a clean one.
  exit "${PIPESTATUS[0]}" ) &
SYNC_PID=$!
echo "$SYNC_PID" > "$STATUS/sync_pid"
log "sync pid $SYNC_PID (under a pty so the progress bar renders)"

# py-spy attaches to two workers once they exist. It cannot wrap them at launch
# the way the env-var harness did, because the branch spawns them itself.
# py-spy needs ptrace. yama's ptrace_scope=1 permits tracing DESCENDANTS ONLY,
# and the native fan-out spawns its own workers, so py-spy can only ever attach
# to them as a sibling. The env-var harness wrapped each worker at launch and was
# therefore the parent; that option is gone with the fan-out. Checked up front
# and recorded, because the alternative is discovering at the end of a six-hour
# run that the profiles were never written. /proc/sys is read-only in a RunPod
# container, so this cannot be fixed from inside one: it needs a pod created with
# CAP_SYS_PTRACE.
PTRACE_SCOPE=$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo 0)
if [ "$PROFILE" = "1" ] && [ "$PTRACE_SCOPE" != "0" ]; then
  PROFILE=0
  echo "ptrace_scope=$PTRACE_SCOPE: py-spy cannot attach to workers it did not spawn" \
    > "$PROF/spy.status"
  log "PROFILING OFF: ptrace_scope=$PTRACE_SCOPE blocks attaching to the fan-out's workers"
fi
if [ "$PROFILE" = "1" ]; then
  # Matched by spawn_main, not by parenthood: the sync now runs behind script and
  # a filter pipeline, so the workers are grandchildren, and lilbee's other
  # children are llama-server processes that py-spy has nothing to say about.
  ( kids=()
    for _ in $(seq 1 60); do
      mapfile -t kids < <(pgrep -f "[s]pawn_main" 2>/dev/null)
      [ "${#kids[@]}" -ge 2 ] && break
      sleep 10
    done
    [ "${#kids[@]}" -ge 2 ] || { echo "no workers to profile" > "$PROF/spy.status"; exit 0; }
    /root/venv/bin/py-spy record --gil --nonblocking --rate "$PROFILE_RATE" --format raw \
      --output "$PROF/w0.gil.folded" --pid "${kids[0]}" >>"$PROF/spy.err" 2>&1 &
    /root/venv/bin/py-spy record --idle --nonblocking --rate "$PROFILE_RATE" --format raw \
      --output "$PROF/w1.wall.folded" --pid "${kids[1]}" >>"$PROF/spy.err" 2>&1 &
    echo "gil=${kids[0]} wall=${kids[1]}" > "$PROF/spy.status" ) &
fi

# Extraction percentiles over EVERY traced file, kept current by byte offset so a
# tick costs the new lines rather than the whole corpus.
( while kill -0 "$SYNC_PID" 2>/dev/null; do
    "$VPY" /root/extract_hist.py "$PROF" > "$PROF/extract.summary" 2>/dev/null
    sleep 30
  done ) &

# Telemetry is small and is the analytical point of the run, so it goes up while
# the run is still going: an unrecoverable pod then still leaves the measurement.
if [ -n "${HF_REPO:-}" ] && [ "${TELEMETRY_TICK:-1}" = "1" ]; then
  ( while kill -0 "$SYNC_PID" 2>/dev/null; do
      sleep 1800
      HF_TOKEN="${HF_TOKEN:-}" TELEMETRY_ONLY=1 bash /root/publish9m.sh >> "$VOL/publish.log" 2>&1
    done ) &
fi

wait "$SYNC_PID"; RC=$?
SECS=$(( $(date -u +%s) - T0 ))
echo "$RC" > "$STATUS/sync_rc"
log "sync rc=$RC after ${SECS}s"
for pid in "${SAMPLERS[@]}"; do kill "$pid" 2>/dev/null; done
pkill -f "[p]y-spy record" 2>/dev/null

# --- what landed --------------------------------------------------------------
phase "counting what landed"
LANDED=$("$VPY" /root/rows_sampler.py "$DATA" - 0 2>/dev/null | tail -1 | cut -d, -f2)
SHARDS=$("$VPY" /root/rows_sampler.py "$DATA" - 0 2>/dev/null | tail -1 | cut -d, -f3-  \
         | tr ',' '\n' | awk '{s+=$1} END {print s+0}')
{
  echo "expected=$EXPECTED"
  echo "landed=$LANDED"
  echo "shard_sources=$SHARDS"
  echo "sync_rc=$RC"
  echo "ingest_secs=$SECS"
  echo "docs_per_s=$("$VPY" -c "print(f'{${LANDED:-0}/max($SECS,1):.1f}')")"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$STATUS/counts"
cat "$STATUS/counts"

if [ "${LANDED:-0}" != "$EXPECTED" ]; then
  # Recorded, not acted on. The artifacts are produced either way; this line is
  # what a reader of the summary needs in order to distrust the row count.
  log "COUNT MISMATCH: landed=$LANDED expected=$EXPECTED (publishing anyway)"
  echo "landed=$LANDED expected=$EXPECTED" > "$VOL/COUNT_MISMATCH"
fi
[ "$RC" = "0" ] || log "SYNC FAILED rc=$RC; shards are kept and a re-run resumes"

phase "merge done, ready to publish"
date -u +%s > "$VOL/MERGE_DONE"
log "index is on the volume. publish9m.sh takes it from here."
