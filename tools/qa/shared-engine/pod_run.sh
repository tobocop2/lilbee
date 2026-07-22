#!/usr/bin/env bash
# Pod-side runbook for the shared-engine harness. Runs inside the pod's tmux.
set -uo pipefail
# Defaults to main so the harness keeps working once this branch merges and is
# deleted; the bootstrap is fetched from a raw URL that would 404 otherwise.
BRANCH="${LILBEE_QA_BRANCH:-main}"
WHEEL_URL="https://github.com/tobocop2/lilbee/releases/download/v0.6.90b420.dev721/lilbee_engine-0.6.90b420.dev721-1.cu124-py3-none-manylinux_2_17_x86_64.whl"
ENGINE_CACHE=/workspace/engine-cache/cu124

log() { echo "[pod-run $(date +%H:%M:%S)] $*"; }

# 1. Pre-seed the engine cache from the prebuilt wheel (never compile on pods).
if [ ! -x "$ENGINE_CACHE/llama-server" ]; then
  log "seeding engine cache from prebuilt cu124 wheel"
  mkdir -p "$ENGINE_CACHE" /tmp/engwheel
  for i in 1 2 3; do
    curl -fL --retry 3 -o /tmp/engwheel/engine.whl "$WHEEL_URL" && break
    log "wheel download attempt $i failed; retrying"
    sleep 5
  done
  python3 -c "import zipfile; zipfile.ZipFile('/tmp/engwheel/engine.whl').extractall('/tmp/engwheel/x')"
  cp -a /tmp/engwheel/x/lilbee_engine/bin/. "$ENGINE_CACHE/"
fi
# zipfile.extractall drops the executable bits; gguf-parser missing +x silently
# degrades planning to file-size estimates.
chmod +x "$ENGINE_CACHE"/llama-* "$ENGINE_CACHE"/gguf-parser 2>/dev/null
ls -la "$ENGINE_CACHE" | head -8

# 2. Bootstrap (clones the branch, uv sync, CUDA runtime libs, opencode pin).
log "fetching bootstrap from the branch"
curl -fsSL "https://raw.githubusercontent.com/tobocop2/lilbee/$BRANCH/tools/qa/opencode/pod_bootstrap.sh" -o /tmp/pod_bootstrap.sh
BRANCH="$BRANCH" bash /tmp/pod_bootstrap.sh 2>&1 | tee /root/bootstrap.log | tail -5 || { log "BOOTSTRAP FAILED"; exit 1; }

# 3. Run the harness.
log "starting harness"
source /root/lilbee_venv/bin/activate
export UV_PROJECT_ENVIRONMENT=/root/lilbee_venv UV_NO_SYNC=1
export PATH="$HOME/.opencode/bin:$PATH"
mkdir -p /root/harness-results
REPO_DIR=/workspace/lilbee bash /workspace/lilbee/tools/qa/shared-engine/harness.sh 2>&1 | tee /root/harness-results/run.log
rc=$?
log "harness exit code: $rc"
echo "$rc" > /root/harness-results/exitcode
