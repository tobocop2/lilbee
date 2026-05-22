#!/usr/bin/env bash
# After a code fix lands on the branch, pull the change and re-run just the
# cells you want (the matrix harness keeps results.md fresh per run).
#
# Usage on the cloud box:
#   bash tools/qa/cloud-rerun.sh                       # all active cells
#   bash tools/qa/cloud-rerun.sh ernie,lfm2,cohere     # targeted families

set -euxo pipefail

BRANCH="${LILBEE_BRANCH:-feat/local-model-api}"
FAMILIES="${1:-}"
WORK_DIR="${HOME}/lilbee"

cd "${WORK_DIR}"
git fetch origin "${BRANCH}"
git reset --hard "origin/${BRANCH}"
uv sync --extra remote --extra crawler --extra graph \
  --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/${LLAMA_CUDA:-cu124}/"

# Drop any stale matrix process / tmux session from the previous run
tmux kill-session -t lilbee-matrix 2>/dev/null || true
pkill -f "tools/qa/opencode/matrix.py" 2>/dev/null || true

# Wipe per-cell logs we're about to overwrite; preserve any cells we're not re-running.
if [ -z "${FAMILIES}" ]; then
  rm -rf tools/qa/opencode/results tools/qa/opencode/logs
  ARGS=()
else
  IFS=',' read -ra FAMS <<<"${FAMILIES}"
  for f in "${FAMS[@]}"; do
    rm -f "tools/qa/opencode/logs/${f}.log"
  done
  ARGS=(--families "${FAMILIES}")
fi

LOG=/tmp/qa-matrix-$(date +%Y%m%d-%H%M%S).log
tmux new-session -d -s lilbee-matrix \
  "cd ${WORK_DIR} && export PATH=${HOME}/.local/bin:${HOME}/.opencode/bin:${PATH} && HF_HUB_DISABLE_PROGRESS_BARS=1 uv run python -u tools/qa/opencode/matrix.py ${ARGS[*]+\"${ARGS[@]}\"} 2>&1 | tee ${LOG}"

cat <<EOF

Re-run kicked off in tmux session 'lilbee-matrix'.

  Stream events:        tail -F ${LOG} | grep -E '^\\['
  Attach foreground:    tmux attach -t lilbee-matrix
  Detach again:         Ctrl-b then d
  When done, results:   ${WORK_DIR}/tools/qa/opencode/results/results.md

EOF
