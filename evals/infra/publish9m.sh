#!/usr/bin/env bash
# Export the finished index off the network volume and publish it to HuggingFace.
#
# Runs on a CHEAP pod attached to the same volume, not on the eight H100s that
# built the index: an upload is network-bound and does not need $21.52/hr of idle
# GPU behind it. That split is what the network volume buys.
#
# TELEMETRY_ONLY=1 makes this a mid-run tick, safe to call repeatedly from the
# GPU pod while the ingest is still going: it refreshes the samplers and profiles
# only, so the analytical part of the run is already published long before the
# index is. Losing the pod then costs the artifacts, never the measurement.
#
# Upload order is smallest first for the same reason. At the full corpus:
#   telemetry ~200MB   dataset ~4GB   index ~150GB
set -uo pipefail

: "${VOL:=/workspace}"
: "${HF_REPO:=beeberg/msmarco-ingest-checkpoint}"
: "${HF_PRIVATE:=1}"
: "${TELEMETRY_ONLY:=0}"
: "${UPLOAD_INDEX:=1}"
: "${EXPORT_PREFIX:=msmarco}"

DATA="$VOL/kb/.lilbee"; PROF="$VOL/prof"; STATUS="$VOL/status"
EXPORT_DIR="$VOL/export"; TELEMETRY="$VOL/telemetry"
VPY=$(cat "$STATUS/vpy" 2>/dev/null || echo /root/venv/bin/python)
LB=$(dirname "$VPY")/lilbee
log()  { printf '[publish %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die()  { log "FATAL: $*"; exit 1; }

[ -n "${HF_TOKEN:-}" ] || die "no HF_TOKEN"
[ -d "$DATA" ] || die "no index at $DATA"

# The venv lives on the container disk, not the volume, so a replacement pod
# reaches the index with no lilbee to read it with. Installing here is what lets
# this run anywhere the volume can be attached, which is the whole reason the
# upload happens on a cheap pod instead of on eight idle H100s.
# No engine wheel: export reads the store and writes text, and nothing on this
# path embeds anything.
if [ ! -x "$VPY" ]; then
  log "no lilbee on this pod; installing (export needs no GPU and no engine)"
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  uv venv --clear --seed --python 3.12 /root/venv || die "venv"
  VPY=/root/venv/bin/python; LB=/root/venv/bin/lilbee
  uv pip install -q --python "$VPY" --prerelease=allow \
    "git+https://github.com/tobocop2/lilbee@${BRANCH:-feat/native-multi-gpu-ingest}" \
    || die "lilbee install"
  uv pip install -q --python "$VPY" huggingface_hub || die "hub install"
  echo "$VPY" > "$STATUS/vpy"
fi

# --- telemetry ----------------------------------------------------------------
log "collecting telemetry"
mkdir -p "$TELEMETRY"
"$VPY" /root/extract_hist.py "$PROF" > "$PROF/extract.summary" 2>/dev/null
"$VPY" /root/summarize.py "$PROF" > "$PROF/SUMMARY.txt" 2>&1 || true
for f in "$PROF"/*.csv "$PROF"/*.folded "$PROF"/*.svg "$PROF"/SUMMARY.txt \
         "$PROF"/extract.summary "$PROF"/spy.status \
         "$STATUS"/run.env "$STATUS"/counts "$STATUS"/config.toml "$VOL"/ingest.log; do
  [ -e "$f" ] && cp -a "$f" "$TELEMETRY/" 2>/dev/null
done
# The trace is the bulk of the telemetry (~1GB at the full corpus) and compresses
# roughly 7x, so it is copied compressed rather than copied then compressed.
if [ -f "$PROF/extract.trace.log" ]; then
  gzip -c "$PROF/extract.trace.log" > "$TELEMETRY/extract.trace.log.gz" 2>/dev/null
fi
# Flame graphs, when the folded stacks made it off the run.
if command -v flamegraph.pl >/dev/null; then
  for folded in "$TELEMETRY"/*.folded; do
    [ -e "$folded" ] && flamegraph.pl "$folded" > "${folded%.folded}.svg" 2>/dev/null
  done
fi

cat > "$TELEMETRY/README.md" <<'TEOF'
# Ingest telemetry

Samplers and profiles from one full-corpus lilbee ingest across 8 GPUs, run
through the native per-GPU fan-out (one bare `lilbee sync`, no environment
variables).

| file | what |
|---|---|
| `SUMMARY.txt` | whole-run CPU/GPU load, power, throughput, extraction percentiles |
| `counts` | expected vs landed passages, sync exit code, ingest seconds, docs/s |
| `run.env` | cards, cores, model, branch, volume backing the index |
| `config.toml` | every setting this run changed from lilbee's defaults |
| `extract.trace.log.gz` | one line per extracted file: source, type, elapsed_ms, pages, chunks |
| `extract.summary` | exact extraction percentiles over every file, not a tail sample |
| `gpu.csv` | ts + per-card GPU utilisation, all 8 cards (2s) |
| `rows.csv` | ts, merged rows, then one column per shard (20s) |
| `host.csv` | ts, load average, total threads, MemAvailable MB (10s) |
| `sys.csv` | ts, total GPU watts, bytes used on the volume (30s) |
| `w0.gil.folded` | py-spy samples holding the GIL, one worker (folded stacks) |
| `w1.wall.folded` | py-spy samples of all wall time, one worker (folded stacks) |
| `spy.status` | whether profiling ran, and why not when it did not |
| `ingest.log` | the run's own log |

The folded files load directly in https://speedscope.app and render with
flamegraph.pl. GIL-held fraction of sampled wall time is
`sum(w0.gil.folded) / sum(w1.wall.folded)`, and the two are different workers,
so read it as a fleet-level ratio rather than one process's.

The folded files are ABSENT when `spy.status` says so. py-spy needs ptrace, and
yama's `ptrace_scope=1` allows tracing descendants only; the native fan-out
spawns its own workers, so py-spy can only reach them as a sibling. The earlier
environment-variable harness wrapped each worker at launch and was therefore its
parent, which is why its profiles exist and these do not. `/proc/sys` is
read-only inside a RunPod container, so it cannot be lifted from within one: the
pod has to be created with CAP_SYS_PTRACE. Every other file here is unaffected.
TEOF
log "  telemetry: $(du -sh "$TELEMETRY" 2>/dev/null | cut -f1)"

# --- dataset ------------------------------------------------------------------
if [ "$TELEMETRY_ONLY" != "1" ]; then
  log "exporting the passage dataset"
  mkdir -p "$EXPORT_DIR"
  for fmt in parquet jsonl; do
    out="$EXPORT_DIR/${EXPORT_PREFIX}-passages.${fmt}"
    if [ -s "$out" ]; then
      log "  ${fmt}: already exported ($(du -h "$out" | cut -f1))"
      continue
    fi
    if LILBEE_DATA="$DATA" "$LB" export "$out" 2>&1 | tail -2; then
      log "  ${fmt}: $(du -h "$out" 2>/dev/null | cut -f1)"
    else
      # Not fatal: the index itself is the primary artifact and must still go up.
      log "  ${fmt}: EXPORT FAILED, continuing"
    fi
  done
  {
    echo "# MS MARCO passage index (lilbee, native per-GPU ingest)"
    echo
    echo "Built by one \`lilbee sync\` across 8 H100s. No per-worker environment"
    echo "variables, no corpus sharding by hand, no merge script: lilbee spawns one"
    echo "worker per card, deals the corpus by predicate and folds the shards itself."
    echo
    echo '```'
    cat "$STATUS/run.env" 2>/dev/null
    cat "$STATUS/counts" 2>/dev/null
    echo '```'
    [ -f "$VOL/COUNT_MISMATCH" ] && {
      echo
      echo "> **Count mismatch.** $(cat "$VOL/COUNT_MISMATCH"). The artifacts are"
      echo "> published as built; treat the row count as the measured one."
    }
    echo
    echo "- \`dataset/\` passage text as parquet and jsonl (no vectors)"
    echo "- \`index/\` a lilbee data root: \`lilbee search --data-dir <dir>\`"
    echo "- \`telemetry/\` samplers, traces and profiles from the run"
    echo "- \`recording/\` asciinema casts of the run's dashboard"
  } > "$EXPORT_DIR/README.md"
fi

# --- upload -------------------------------------------------------------------
log "uploading to $HF_REPO (telemetry_only=$TELEMETRY_ONLY index=$UPLOAD_INDEX)"
HF_REPO="$HF_REPO" HF_PRIVATE="$HF_PRIVATE" TELEMETRY="$TELEMETRY" \
EXPORT_DIR="$EXPORT_DIR" INDEX_ROOT="$DATA" TELEMETRY_ONLY="$TELEMETRY_ONLY" \
UPLOAD_INDEX="$UPLOAD_INDEX" "$VPY" - <<'PYEOF'
import os
import time

from huggingface_hub import HfApi

repo = os.environ["HF_REPO"]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(
    repo_id=repo, repo_type="dataset",
    private=os.environ.get("HF_PRIVATE", "1") == "1", exist_ok=True,
)


failures: list[str] = []

# Errors that no amount of retrying will fix. A storage quota is the one that
# matters here: retrying it burns a paid pod against a wall and, worse, the
# partial upload left behind is a CORRUPT index rather than a smaller one.
_TERMINAL = ("storage limit", "quota", "402", "payment required")


def push(folder: str, path_in_repo: str, message: str, *, ignore=None) -> None:
    """Upload *folder*, retrying: a six-hour run must not lose its artifacts to one 502."""
    if not folder or not os.path.isdir(folder):
        return
    for attempt in range(1, 6):
        try:
            api.upload_folder(
                repo_id=repo, repo_type="dataset", folder_path=folder,
                path_in_repo=path_in_repo, commit_message=message,
                ignore_patterns=ignore,
            )
            print(f"[publish] {path_in_repo}/ uploaded", flush=True)
            return
        except Exception as exc:  # noqa: BLE001 - any transport failure is retryable here
            print(f"[publish] {path_in_repo}/ attempt {attempt} failed: {exc}", flush=True)
            if any(t in str(exc).lower() for t in _TERMINAL):
                print(f"[publish] {path_in_repo}/ FAILED TERMINALLY: not retryable", flush=True)
                failures.append(path_in_repo)
                return
            time.sleep(30 * attempt)
    print(f"[publish] {path_in_repo}/ GAVE UP after 5 attempts", flush=True)
    failures.append(path_in_repo)


push(os.environ["TELEMETRY"], "telemetry", "ingest telemetry")
if os.environ.get("TELEMETRY_ONLY") != "1":
    push(os.environ["EXPORT_DIR"], "dataset", "passage dataset: parquet + jsonl")
    if os.environ.get("UPLOAD_INDEX", "1") == "1":
        print("[publish] uploading the index; ~150GB at the full corpus", flush=True)
        # The shards are the resume state, not a deliverable: they hold a second
        # copy of every vector already in the merged index.
        push(os.environ["INDEX_ROOT"], "index", "lilbee index: vectors, ANN and FTS",
             ignore=["shards/**"])
PYEOF
rc=$?
[ "$rc" = "0" ] || die "upload rc=$rc"
[ "$TELEMETRY_ONLY" = "1" ] || date -u +%s > "$VOL/PUBLISH_DONE"
log "DONE: https://huggingface.co/datasets/$HF_REPO"
