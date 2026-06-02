#!/usr/bin/env bash
# Fresh-pod build stage for the opencode model reel (RunPod 2xH200, ephemeral /root).
#
# This is the EXACT model-independent bootstrap that runs first on a fresh pod:
# system deps -> uv -> clone lilbee -> uv sync -> build a CUDA llama-server from
# source (no prebuilt CUDA Linux binary ships in ggml-org releases). It writes
# progress to /root/run.log and ends with the BUILD_STAGE_DONE marker.
#
# It is launched detached inside the `work` tmux session so it survives SSH drops
# and so the run can be monitored with `tmux attach -t work` or `tail -F /root/run.log`.
#
# After this completes, the per-model serve/index/record pipeline runs:
#   giant_demo.sh  -> warm a giant on llama-server + index a codebase + lilbee serve /mcp
#   build_reel.sh  -> record the live opencode TUI tape + cold-start card + review gate
#
# Everything runs on the pod, including VHS recording. The "VHS captures 0 frames on
# the pod" failure was an outdated ttyd (apt ships 1.6.3; VHS needs >=1.7.2) plus the
# absolute-Output-path parser bug -- both fixed in the VHS step below. No SSH tunnel,
# no Mac, no public exposure: opencode + VHS record the live TUI locally on the box.
set -uo pipefail
LOG=/root/run.log
exec >>"$LOG" 2>&1
ts(){ date -u +%H:%M:%S; }
step(){ echo "[$(ts)] === STEP: $* ==="; }
fail(){ echo "[$(ts)] !!! FAIL: $* !!!"; echo "BOOTSTRAP_FAILED"; exit 1; }
export DEBIAN_FRONTEND=noninteractive
export PATH=$HOME/.local/bin:$PATH

step "apt build deps"
apt-get update -qq || fail "apt update"
apt-get install -y -qq cmake build-essential pkg-config ccache tmux >/dev/null || fail "apt install"
echo "[$(ts)] apt OK"

step "install uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh || fail "uv install"
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || fail "uv missing after install"
echo "[$(ts)] uv $(uv --version)"

step "clone lilbee (public)"
cd /root
[ -d /root/lilbee ] || git clone -q https://github.com/tobocop2/lilbee.git || fail "clone lilbee"
cd /root/lilbee
git fetch -q origin || true
git checkout feat/local-model-api || fail "checkout branch"
echo "[$(ts)] lilbee @ $(git rev-parse --short HEAD)"

step "uv sync lilbee (slow)"
uv sync || fail "uv sync"
echo "[$(ts)] uv sync OK"

step "clone + build CUDA llama-server (long pole ~10-20min)"
cd /root
[ -d /root/llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp || fail "clone llama.cpp"
cd /root/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF || fail "cmake configure"
echo "[$(ts)] cmake configured, compiling llama-server..."
cmake --build build -j --target llama-server || fail "cmake build"
test -f build/bin/llama-server || fail "llama-server binary missing"
ln -sf /root/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
echo "[$(ts)] llama-server: $(command -v llama-server)"

step "install VHS recorder (ffmpeg + ttyd>=1.7.2 + vhs + headless-chromium deps)"
apt-get install -y -qq ffmpeg >/dev/null || fail "apt ffmpeg"
# apt's ttyd is 1.6.3 which VHS rejects ("ttyd out of date, VHS requires 1.7.2");
# install the release binary. This is the fix that makes VHS record on the pod.
curl -fsSL https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 -o /usr/local/bin/ttyd \
  && chmod +x /usr/local/bin/ttyd || fail "ttyd binary"
hash -r
mkdir -p /etc/apt/keyrings
curl -fsSL https://repo.charm.sh/apt/gpg.key | gpg --dearmor -o /etc/apt/keyrings/charm.gpg
echo "deb [signed-by=/etc/apt/keyrings/charm.gpg] https://repo.charm.sh/apt/ * *" > /etc/apt/sources.list.d/charm.list
apt-get update -qq && apt-get install -y -qq vhs >/dev/null || fail "apt vhs"
# go-rod (VHS's headless chromium) runtime libs.
apt-get install -y -qq libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
  libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 \
  libpango-1.0-0 libcairo2 libatspi2.0-0 libxshmfence1 >/dev/null || true
apt-get install -y -qq libasound2 >/dev/null 2>&1 || apt-get install -y -qq libasound2t64 >/dev/null 2>&1 || true
echo "[$(ts)] vhs=$(vhs --version 2>&1) ttyd=$(ttyd --version 2>&1)"
# Smoke-test VHS now so a regression is caught at bootstrap, not at record time.
# NOTE: the Output path MUST be relative -- an absolute path trips VHS's parser.
( cd /root && printf 'Output vhscheck.gif\nSet Width 800\nSet Height 400\nType "echo vhs-ok"\nEnter\nSleep 1s\n' > vhscheck.tape \
  && VHS_NO_SANDBOX=true vhs vhscheck.tape >/dev/null 2>&1 \
  && [ "$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 vhscheck.gif 2>/dev/null)" -gt 1 ] \
  && echo "[$(ts)] VHS smoke-test PASS (frames rendered)" \
  || echo "[$(ts)] VHS smoke-test FAIL -- try: xvfb-run VHS_NO_SANDBOX=true vhs vhscheck.tape" )

step "install opencode CLI (recorded on the pod by build_reel.sh)"
# Official opencode org is anomalyco/opencode (moved from sst); opencode.ai/install
# resolves here. Linux CLI ships as a tar.gz, not a zip.
apt-get install -y -qq unzip >/dev/null 2>&1 || true
curl -fsSL https://github.com/anomalyco/opencode/releases/latest/download/opencode-linux-x64.tar.gz -o /root/opencode.tar.gz || fail "opencode download"
tar -xzf /root/opencode.tar.gz -C /usr/local/bin/ || fail "opencode extract"
chmod +x /usr/local/bin/opencode; hash -r
echo "[$(ts)] opencode=$(command -v opencode) ver=$(opencode --version 2>&1 | head -1)"

mkdir -p /root/models
echo "[$(ts)] ====== BUILD_STAGE_DONE ======"
