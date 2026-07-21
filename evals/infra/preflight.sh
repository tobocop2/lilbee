#!/usr/bin/env bash
# Prove the box can do the work BEFORE the work starts.
#
# Every check here exists because skipping it cost real hours on a previous run.
# An 8.8M-passage ingest is a multi-hour, billed operation; discovering in hour
# three that the engine has no kernels for this GPU, or that CUDA is broken in a
# way nvidia-smi does not report, means paying for the discovery twice.
#
# Fails loudly and early. Never "warn and continue": a warning at the top of a
# four-hour job is a warning nobody reads.
set -euo pipefail

log() { printf '[preflight %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { printf '[preflight FAIL] %s\n' "$*" >&2; exit 1; }

log "GPUs as the driver reports them"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap \
           --format=csv,noheader || die "nvidia-smi failed; this box has no usable driver"

# Community pods routinely arrive with nvidia-smi working and cuInit broken.
# torch is the cheapest thing that actually initialises a CUDA context.
log "CUDA context actually initialises (nvidia-smi alone does not prove this)"
"$PYBIN" - <<'PY' || die "CUDA present per nvidia-smi but unusable; reclaim this pod"
import sys, torch
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False")
n = torch.cuda.device_count()
for i in range(n):
    cap = torch.cuda.get_device_capability(i)
    print(f"  gpu{i} {torch.cuda.get_device_name(i)} sm_{cap[0]}{cap[1]}")
# Touch each card: allocation is what fails when a context is broken.
for i in range(n):
    torch.zeros(1024, 1024, device=f"cuda:{i}").sum().item()
print(f"  all {n} GPU(s) allocated and computed")
PY

# A prebuilt engine is a release artifact, not a compile: if it is missing the
# only sane response is to stop, because the fallback is an eight-hour build on
# a GPU box.
log "engine binary is present and prebuilt (never compiled here)"
"$PYBIN" -c 'import lilbee_engine, os, sys; p = lilbee_engine.get_llama_server_path(); sys.exit(0 if os.access(p, os.X_OK) else f"engine missing or not executable at {p}")' \
  || die "no usable llama-server; the engine wheel is a stub or was not installed"

# Whether the shipped engine covers this card is checked against the card, not
# assumed from a note. An earlier comment asserted the cu124 build had no sm_90
# kernels; the build pins 70;75;80;86;89;90, so it does. Ask the box.
log "engine has kernels for this GPU's compute capability"
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
log "  compute capability sm_${CAP}, engine backend ${BACKEND:-cu124}"
"$LILBEE_BIN" models doctor 2>&1 | tee /tmp/engine_doctor.txt || true
if grep -qiE "no CUDA-capable device|no kernel image" /tmp/engine_doctor.txt; then
  die "engine has no kernels for sm_${CAP}; pick a GPU the ${BACKEND:-cu124} build covers"
fi

log "embedding model actually loads and returns a vector"
"$LILBEE_BIN" models pull "${EMBED_MODEL}" 2>&1 | tail -3
"$PYBIN" - <<'PY' || die "the embedder did not return a usable vector; the ingest would produce an empty index"
import os, sys, httpx, time
base = os.environ["LILBEE_BASE_URL"].rstrip("/")
for attempt in range(60):
    try:
        r = httpx.post(f"{base}/v1/embeddings",
                       json={"model": os.environ["EMBED_MODEL"], "input": "preflight probe"},
                       timeout=120)
        r.raise_for_status()
        vec = r.json()["data"][0]["embedding"]
        print(f"  embedding dim={len(vec)}")
        if len(vec) < 8:
            sys.exit(f"embedder returned a degenerate vector of length {len(vec)}")
        break
    except Exception as exc:
        if attempt == 59:
            sys.exit(f"embedder never came up: {exc}")
        time.sleep(10)
PY

log "disk headroom for the index"
AVAIL=$(df -BG --output=avail "$WORK_DIR" | tail -1 | tr -dc '0-9')
log "  ${AVAIL}G available at $WORK_DIR"
[ "$AVAIL" -ge "${MIN_DISK_GB:-120}" ] || die "only ${AVAIL}G free at $WORK_DIR; need ${MIN_DISK_GB:-120}G for 8.8M vectors plus working space"

log "trace logging is switched on and writable"
[ "${LILBEE_INGEST_TRACE:-}" = "1" ] || die "LILBEE_INGEST_TRACE is not 1; the run would produce no per-document timings"
: > "${LILBEE_INGEST_TRACE_FILE:?LILBEE_INGEST_TRACE_FILE unset}" \
  || die "cannot write the trace file at ${LILBEE_INGEST_TRACE_FILE}"

log "ALL CHECKS PASSED - safe to start the ingest"
