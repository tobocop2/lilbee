#!/usr/bin/env bash
# One-shot setup for an Ubuntu+CUDA GPU cloud box (Lambda Labs, RunPod, Vast.ai, ...).
#
# Fresh SSH session -> running QA matrix in one command.
#
# Usage on the cloud box:
#   bash <(curl -fsSL https://raw.githubusercontent.com/tobocop2/lilbee/feat/local-model-api/tools/qa/cloud-setup.sh)
#
# Or if you've already cloned the repo:
#   cd lilbee && bash tools/qa/cloud-setup.sh
#
# Optional env:
#   LILBEE_BRANCH=feat/local-model-api  (default)
#   LLAMA_CUDA=cu124                    (cu124|cu125|cu126 - pick to match the box's nvidia-smi CUDA version)
#   HF_TOKEN=hf_xxx                     (only if pulling gated repos; the default matrix cells are all public)

set -euxo pipefail

BRANCH="${LILBEE_BRANCH:-feat/local-model-api}"
CUDA="${LLAMA_CUDA:-cu124}"
REPO_URL="https://github.com/tobocop2/lilbee.git"
WORK_DIR="${HOME}/lilbee"

# 1. System deps (Lambda's PyTorch image already has most of these; idempotent)
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  git tmux curl ca-certificates build-essential pkg-config jq

# 2. Install uv (Python toolchain) if not already on PATH
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

# 3. Install opencode (the agent client the matrix drives)
if ! command -v opencode >/dev/null 2>&1; then
  curl -fsSL https://opencode.ai/install | bash
  export PATH="${HOME}/.opencode/bin:${PATH}"
fi

# 4. Clone or update lilbee on the requested branch
if [ ! -d "${WORK_DIR}/.git" ]; then
  git clone "${REPO_URL}" "${WORK_DIR}"
fi
cd "${WORK_DIR}"
git fetch origin "${BRANCH}"
git checkout "${BRANCH}"
git reset --hard "origin/${BRANCH}"

# 5. Install lilbee with the [remote] + [crawler] + [graph] extras + CUDA wheel.
# Sync via PyPI first (resolves the dep tree), then force-reinstall the GPU
# build of llama-cpp-python from abetlen's CUDA-specific index. The two-step
# is required because passing the CUDA URL as ``--extra-index-url`` only
# ranks it BELOW PyPI; uv would still pick PyPI's CPU wheel for the same
# version. ``--index-url`` (primary) on the second call forces the GPU build.
uv sync --extra remote --extra crawler --extra graph
uv pip install --reinstall-package llama-cpp-python --no-cache \
  --index-url "https://abetlen.github.io/llama-cpp-python/whl/${CUDA}/" \
  llama-cpp-python

# Sanity-check: confirm GPU offload is actually compiled in.
uv run python -c "from llama_cpp.llama_cpp import llama_supports_gpu_offload; assert llama_supports_gpu_offload(), 'CUDA wheel missing -- llama_cpp built without GPU offload'; print('GPU offload OK')"

# 6. Unskip the GPU-enabled cells (these were skipped on the user's M1 Pro)
uv run python - <<'PYEOF'
import re
from pathlib import Path

p = Path("tools/qa/opencode/models.toml")
src = p.read_text()

# Drop the trailing ``skip = true ...\n`` line on glm-air + phi4mini blocks.
# Leave mistral v0.3 + gemma2 skipped (not tool-trained, lilbee correctly refuses).
for family in ("glm-air", "phi4mini"):
    pattern = re.compile(
        r'(\[\[model\]\]\nfamily = "' + re.escape(family) + r'".*?)\nskip = true[^\n]*',
        re.DOTALL,
    )
    new = pattern.sub(r"\1", src)
    if new == src:
        print(f"WARN: {family} was already unskipped or pattern missed")
    src = new

p.write_text(src)
print("models.toml updated: glm-air + phi4mini are now active")
PYEOF

# 7. HF env: suppress progress bars so the matrix log stays grep-able
export HF_HUB_DISABLE_PROGRESS_BARS=1
if [ -n "${HF_TOKEN:-}" ]; then
  uv run huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential
fi

# 8. Run the matrix in a detached tmux session so SSH disconnects don't kill it
LOG=/tmp/qa-matrix-$(date +%Y%m%d-%H%M%S).log
tmux kill-session -t lilbee-matrix 2>/dev/null || true
tmux new-session -d -s lilbee-matrix \
  "cd ${WORK_DIR} && export PATH=${HOME}/.local/bin:${HOME}/.opencode/bin:${PATH} && HF_HUB_DISABLE_PROGRESS_BARS=1 uv run python -u tools/qa/opencode/matrix.py 2>&1 | tee ${LOG}"

cat <<EOF

QA matrix kicked off in tmux session 'lilbee-matrix'.

  Attach (foreground):  tmux attach -t lilbee-matrix
  Detach from tmux:     Ctrl-b then d
  Stream cell events:   tail -F ${LOG} | grep -E '^\\['
  When done, results:   ${WORK_DIR}/tools/qa/opencode/results/results.md

You can safely log out; the matrix keeps running.
EOF
