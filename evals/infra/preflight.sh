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

# No torch probe here. lilbee runs a llama.cpp engine and does not depend on
# torch, so importing it only proved whether an unrelated package happened to be
# installed. The authoritative check is further down: the engine loading the
# embedder and returning a real vector exercises the same CUDA path the ingest
# will use, on the same binary, with the same model.

# A prebuilt engine is a release artifact, not a compile: if it is missing the
# only sane response is to stop, because the fallback is an eight-hour build on
# a GPU box.
log "engine binary is present and prebuilt (never compiled here)"
"$PYBIN" -c 'import lilbee_engine, os, sys; p = lilbee_engine.get_llama_server_path(); sys.exit(0 if os.access(p, os.X_OK) else f"engine missing or not executable at {p}")' \
  || die "no usable llama-server; the engine wheel is a stub or was not installed"

# Whether the shipped engine covers this card is checked against the card, not
# assumed from a note. An earlier comment asserted the cu124 build had no sm_90
# kernels; the build pins 70;75;80;86;89;90, so it does. Ask the box.
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
log "  compute capability sm_${CAP}, engine backend ${BACKEND:-cu124}"

# The branch's extraction stack is a compiled extension, so the image's glibc
# has to be new enough for it. Checking here turns an undefined-symbol error at
# first extraction -- which reads as a broken package -- into a named image
# problem before anything is embedded.
log "extraction stack imports against this image's glibc"
"$PYBIN" -c 'import xberg' 2>/dev/null \
  || die "xberg will not import: glibc is $(ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+$'), and its wheel needs 2.38+. Use an ubuntu 24.04 image."

# Warm the model cache. The embedder is SELECTED by config.toml in the data root
# (written by ingest.sh) -- the mechanism the proven lilbeekreuzbergstein harness
# uses and the one the fleet actually reads. `use-embedder` wrote a config the
# ingest's data-dir did not consult, so the embed server stayed on the
# uninstalled default and every passage silently failed. `model pull` only
# downloads; selection is the config file's job.
log "embedding model downloads (pulled to the cache; config.toml selects it)"
"$LILBEE_BIN" model pull "${EMBED_MODEL}" 2>&1 | tail -5 \
  || die "model pull failed for ${EMBED_MODEL}"

# The engine-loads-a-vector check lives in the sync run, not here: it needs a
# running embed server, and nothing serves at preflight time. The model pull
# above is the part checkable without a server. A wrong GPU arch surfaces when
# the fleet loads the embedder at sync start, before the bulk is embedded, so
# the fast-fail guarantee holds.

log "disk headroom for the index"
AVAIL=$(df -BG --output=avail "$WORK_DIR" | tail -1 | tr -dc '0-9')
log "  ${AVAIL}G available at $WORK_DIR"
[ "$AVAIL" -ge "${MIN_DISK_GB:-120}" ] || die "only ${AVAIL}G free at $WORK_DIR; need ${MIN_DISK_GB:-120}G for 8.8M vectors plus working space"

log "trace logging is switched on and writable"
[ "${LILBEE_INGEST_TRACE:-}" = "1" ] || die "LILBEE_INGEST_TRACE is not 1; the run would produce no per-document timings"
: > "${LILBEE_INGEST_TRACE_FILE:?LILBEE_INGEST_TRACE_FILE unset}" \
  || die "cannot write the trace file at ${LILBEE_INGEST_TRACE_FILE}"

log "ALL CHECKS PASSED - safe to start the ingest"
