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
  # The cached clone is shallow + single-branch, so a bare `fetch origin` won't
  # pull a branch it wasn't cloned with; fetch the target branch explicitly.
  git -C "$REPO_DIR" fetch --depth 1 origin "$BRANCH" -q
  git -C "$REPO_DIR" checkout -q -B "$BRANCH" FETCH_HEAD
fi
cd "$REPO_DIR"

# 5. Sync into a LOCAL-disk venv (network FS stale-handles + can't hardlink).
log "uv sync -> $UV_PROJECT_ENVIRONMENT"
uv sync

# 6. Engine: build once, cache on /workspace, copy into the venv on every boot.
VENV_ENGINE_BIN="$("$VENV_PY" -c 'import lilbee_engine,os;print(os.path.dirname(lilbee_engine.__file__))')/bin"
# The CUDA toolkit provides BOTH nvcc (for the build) and the runtime libs
# (libcudart/libcublas) that llama-server links at RUNTIME. A fresh container has
# neither, and a cached engine skips the build below -- so this must run on every
# cu* boot when the runtime lib is absent, not only when building. Skipping it on a
# cached-engine pod leaves llama-server dying with "libcudart.so.12: cannot open
# shared object file" on every serve, which fails the whole matrix.
if [[ "$BACKEND" == cu* ]] && ! ldconfig -p 2>/dev/null | grep -q 'libcudart\.so\.12'; then
  log "installing CUDA toolkit (nvcc + runtime libs) for $BACKEND"
  : > /tmp/toolkit-env.sh
  TOOLKIT_ENV_FILE=/tmp/toolkit-env.sh BACKEND="$BACKEND" bash tools/wheel-build/install_gpu_toolkit.sh
  # GitHub Actions propagates the toolkit's location to later steps via $GITHUB_ENV;
  # on a pod there is no such boundary, so source it ourselves (nvcc on PATH) and
  # refresh the linker cache so the engine resolves libcudart at runtime.
  source /tmp/toolkit-env.sh
  # The toolkit drops the runtime libs under /usr/local/cuda*/targets/.../lib but
  # does not add that dir to the linker search path, so `ldconfig` alone never
  # indexes libcudart.so.12 and a cached-engine boot dies with "cannot open
  # libcudart.so.12". Register the dir in ld.so.conf so the engine resolves it
  # system-wide (no per-process LD_LIBRARY_PATH needed).
  CUDA_LIB_DIR=$(ls -d /usr/local/cuda*/targets/x86_64-linux/lib 2>/dev/null | head -1)
  [ -n "$CUDA_LIB_DIR" ] && echo "$CUDA_LIB_DIR" > /etc/ld.so.conf.d/cuda-lilbee.conf
  ldconfig
fi
if [ ! -x "$ENGINE_CACHE/llama-server" ]; then
  log "building engine (BACKEND=$BACKEND) -- one-time ~20min, then cached"
  BACKEND="$BACKEND" bash tools/wheel-build/build_llama_server.sh
  mkdir -p "$ENGINE_CACHE"
  cp -a packaging/engine-wheel/lilbee_engine/bin/. "$ENGINE_CACHE/"
fi
log "installing cached engine into venv"
mkdir -p "$VENV_ENGINE_BIN"
cp -a "$ENGINE_CACHE/." "$VENV_ENGINE_BIN/"

# 7. opencode, pinned (the matrix drives it; standalone install needs no node).
#    A mid-matrix self-update changes the binary under test; the per-cell
#    workspace config also sets autoupdate=false as the second lock.
OPENCODE_PIN="${OPENCODE_PIN:-v1.17.1}"
if [ ! -x "$HOME/.opencode/bin/opencode" ] && ! command -v opencode >/dev/null; then
  log "installing opencode"
  curl -fsSL https://opencode.ai/install | bash
fi
log "pinning opencode to $OPENCODE_PIN"
PATH="$HOME/.opencode/bin:$PATH" opencode upgrade "$OPENCODE_PIN"

# 8. VHS recorder for the demo reels (the cell pane is recorded on the pod, not a Mac).
#    The "VHS captures 0 frames on the pod" block was two mundane causes, both handled
#    here: apt's ttyd is 1.6.3 but VHS needs >=1.7.2 (install the 1.7.7 release binary),
#    and an absolute `Output` path trips VHS's parser (the smoke tape uses a relative
#    Output run from its own dir). VHS also needs VHS_NO_SANDBOX=true on RunPod (user
#    namespaces are disabled) and the go-rod headless-chromium runtime libs.
if ! command -v vhs >/dev/null; then
  log "installing VHS recorder (ttyd>=1.7.2 + vhs + headless-chromium libs)"
  curl -fsSL https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 \
    -o /usr/local/bin/ttyd && chmod +x /usr/local/bin/ttyd
  mkdir -p /etc/apt/keyrings
  curl -fsSL https://repo.charm.sh/apt/gpg.key | gpg --dearmor -o /etc/apt/keyrings/charm.gpg
  echo "deb [signed-by=/etc/apt/keyrings/charm.gpg] https://repo.charm.sh/apt/ * *" \
    > /etc/apt/sources.list.d/charm.list
  apt-get update -qq
  apt-get install -y -qq vhs
  apt-get install -y -qq libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 \
    libpango-1.0-0 libcairo2 libatspi2.0-0 libxshmfence1 || true
  apt-get install -y -qq libasound2 2>/dev/null || apt-get install -y -qq libasound2t64 2>/dev/null || true
fi

# 8b. Smoke-test VHS now so a regression is caught at bootstrap, not at record time
#     (relative Output + VHS_NO_SANDBOX; a frame count > 1 means the pipeline works).
log "VHS smoke-test"
( cd /tmp && printf 'Output vhscheck.gif\nSet Width 800\nSet Height 400\nType "echo vhs-ok"\nEnter\nSleep 1s\n' > vhscheck.tape \
  && VHS_NO_SANDBOX=true vhs vhscheck.tape >/dev/null 2>&1 \
  && [ "$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 vhscheck.gif 2>/dev/null)" -gt 1 ] \
  && log "VHS smoke-test PASS (frames rendered)" \
  || log "VHS smoke-test FAIL -- retry with: xvfb-run VHS_NO_SANDBOX=true vhs vhscheck.tape" )

# 9. Verify the engine resolves from the bundled wheel (no LD_LIBRARY_PATH hacks).
log "verifying engine resolution"
"$VENV_PY" -c "
from lilbee.providers.fleet.binary import resolve_llama_server, resolve_llama_swap, resolve_gguf_parser
for f in (resolve_llama_server, resolve_llama_swap, resolve_gguf_parser):
    print('  engine:', f())
"

# 10. Write a complete, sourceable env file. Printing partial exports left every
#     shell to re-derive the environment and rediscover the non-obvious bits --
#     UV_PROJECT_ENVIRONMENT/UV_NO_SYNC (without which the matrix's `uv run lilbee`
#     re-syncs and the engine-wheel rebuild fails), the persistent models/HF dirs,
#     and the corpus path. One `source` now sets all of it.
log "writing $WORKSPACE/qa_env.sh"
cat > "$WORKSPACE/qa_env.sh" <<EOF
source $UV_PROJECT_ENVIRONMENT/bin/activate
export PATH=/usr/local/go/bin:\$HOME/.local/bin:\$HOME/.opencode/bin:\$PATH
export UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT
export UV_CACHE_DIR=$UV_CACHE_DIR
export UV_LINK_MODE=copy
export UV_NO_SYNC=1
export LILBEE_MODELS_DIR=$LILBEE_MODELS_DIR
export HF_HOME=$HF_HOME
export LILBEE_QA_CORPUS=\${LILBEE_QA_CORPUS:-$WORKSPACE/godot_corpus}
# export HF_TOKEN=...   # set per shell for model pulls; never commit it
EOF

cat <<EOF
[bootstrap] ready. Source the generated env, then run the matrix:
  source $WORKSPACE/qa_env.sh
  export HF_TOKEN=...                       # for model pulls
  python tools/qa/opencode/matrix.py --families <fam>
Recording reels: run VHS as 'VHS_NO_SANDBOX=true vhs <tape>' with a RELATIVE Output
path, from the output dir (an absolute Output trips VHS's parser).
EOF
