#!/usr/bin/env bash
# Bring a fresh from-source pod (RunPod or any Linux + NVIDIA box) to a working
# `lilbee` for the opencode QA matrix. Idempotent: safe to re-run, and a stop/
# resume only repeats the cheap steps.
#
# This script exists because a source checkout is NOT a `pip install lilbee`:
# every gotcha below cost a manual round-trip the first time, so they live here.
#
#   - A RunPod stop/resume gives a FRESH container: anything under /root is wiped,
#     only /workspace (the network volume) survives. So the expensive artifacts
#     (repo, models, HF cache, the built engine) live on /workspace; the cheap
#     toolchain (apt pkgs, Go, uv, opencode) is reinstalled on each fresh boot.
#   - A source checkout ships `packaging/engine-wheel/lilbee_engine/bin/` EMPTY
#     (a `pip install` would pull the prebuilt per-platform lilbee-engine wheel).
#     So the engine is built here via tools/wheel-build and the built binaries are
#     cached on /workspace, so a resume copies them in instead of recompiling ~20min.
#   - The venv on the /workspace network FS hits "Stale file handle" + can't
#     hardlink, so the venv goes on LOCAL disk with UV_LINK_MODE=copy.
#   - System tools a fresh container lacks but the build/harness need:
#     cmake ninja-build build-essential ccache go tmux `time` ffmpeg uv opencode.
#   - lilbee reads LILBEE_DATA / LILBEE_MODELS_DIR (not *_DIR for data); models go
#     to a persistent dir so pulls survive a resume.
set -euo pipefail

BACKEND="${BACKEND:-cu124}"
WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/lilbee}"
BRANCH="${BRANCH:-feat/local-model-api}"
ENGINE_CACHE="$WORKSPACE/engine-cache/$BACKEND"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/root/lilbee_venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/uvcache}"
export UV_LINK_MODE=copy
export LILBEE_MODELS_DIR="${LILBEE_MODELS_DIR:-$WORKSPACE/models}"
export HF_HOME="${HF_HOME:-$WORKSPACE/hf}"
VENV_PY="$UV_PROJECT_ENVIRONMENT/bin/python"

log() { echo "[bootstrap] $*"; }

# 1. System packages: build toolchain + matrix harness (tmux) + timing/reels.
log "apt packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq cmake ninja-build build-essential ccache git curl tmux time ffmpeg

# 2. Go toolchain for the engine helpers (llama-swap, gguf-parser).
if [ ! -x /usr/local/go/bin/go ] && ! command -v go >/dev/null; then
  log "installing Go"
  curl -fsSL https://go.dev/dl/go1.23.4.linux-amd64.tar.gz | tar -C /usr/local -xz
fi
export PATH="/usr/local/go/bin:$PATH"

# 3. uv.
if [ ! -x "$HOME/.local/bin/uv" ] && ! command -v uv >/dev/null; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# 4. Repo on the persistent volume.
if [ ! -d "$REPO_DIR/.git" ]; then
  log "cloning lilbee ($BRANCH)"
  git clone --depth 1 --branch "$BRANCH" https://github.com/tobocop2/lilbee.git "$REPO_DIR"
else
  log "updating lilbee ($BRANCH)"
  git -C "$REPO_DIR" fetch origin -q
  git -C "$REPO_DIR" checkout -q "$BRANCH"
  git -C "$REPO_DIR" reset --hard "origin/$BRANCH"
fi
cd "$REPO_DIR"

# 5. Sync into a LOCAL-disk venv (network FS stale-handles + can't hardlink).
log "uv sync -> $UV_PROJECT_ENVIRONMENT"
uv sync

# 6. Engine: build once, cache on /workspace, copy into the venv on every boot.
VENV_ENGINE_BIN="$("$VENV_PY" -c 'import lilbee_engine,os;print(os.path.dirname(lilbee_engine.__file__))')/bin"
if [ ! -x "$ENGINE_CACHE/llama-server" ]; then
  if [[ "$BACKEND" == cu* ]] && ! command -v nvcc >/dev/null; then
    log "installing CUDA build toolkit for $BACKEND"
    BACKEND="$BACKEND" bash tools/wheel-build/install_gpu_toolkit.sh
  fi
  log "building engine (BACKEND=$BACKEND) -- one-time ~20min, then cached"
  BACKEND="$BACKEND" bash tools/wheel-build/build_llama_server.sh
  mkdir -p "$ENGINE_CACHE"
  cp -a packaging/engine-wheel/lilbee_engine/bin/. "$ENGINE_CACHE/"
fi
log "installing cached engine into venv"
mkdir -p "$VENV_ENGINE_BIN"
cp -a "$ENGINE_CACHE/." "$VENV_ENGINE_BIN/"

# 7. opencode (the matrix drives it; standalone install needs no node).
if [ ! -x "$HOME/.opencode/bin/opencode" ] && ! command -v opencode >/dev/null; then
  log "installing opencode"
  curl -fsSL https://opencode.ai/install | bash
fi

# 8. Verify the engine resolves from the bundled wheel (no LD_LIBRARY_PATH hacks).
log "verifying engine resolution"
"$VENV_PY" -c "
from lilbee.providers.fleet.binary import resolve_llama_server, resolve_llama_swap, resolve_gguf_parser
for f in (resolve_llama_server, resolve_llama_swap, resolve_gguf_parser):
    print('  engine:', f())
"

cat <<EOF
[bootstrap] ready. For each shell:
  source $UV_PROJECT_ENVIRONMENT/bin/activate
  export PATH=/usr/local/go/bin:\$HOME/.local/bin:\$HOME/.opencode/bin:\$PATH
  export LILBEE_MODELS_DIR=$LILBEE_MODELS_DIR HF_HOME=$HF_HOME
Then: lilbee model pull <ref>; python tools/qa/opencode/matrix.py --families <fam>
EOF
