#!/usr/bin/env bash
# Automated QA: proves the native GGUF reranker materially changes the
# chunks cited on concept and entity pages. Runs two lilbee wiki builds
# against the same 5-page PDF slice, one with the reranker disabled
# and one with it on, then compares page sets and citation churn.
#
# Usage:
#   SOURCE_PDF=~/Downloads/cv-manual.pdf ./scripts/qa-wiki-rerank.sh
#
# Required env:
#   SOURCE_PDF               path to an input PDF whose first 5 pages will
#                            drive the comparison.
# Optional env:
#   RERANKER_MODEL           HF slug for the GGUF reranker to pull
#                            (default: CompendiumLabs/bge-reranker-v2-m3-gguf)
#   RERANKER_FILE            filename within the HF repo
#                            (default: bge-reranker-v2-m3-Q4_K_M.gguf)
#   QA_KEEP                  set to 1 to skip teardown of the tmux session
#                            and temp LILBEE_DATA dir for inspection.

set -euo pipefail

: "${SOURCE_PDF:?SOURCE_PDF must point to the input PDF}"
if [[ ! -f "$SOURCE_PDF" ]]; then
  echo "SOURCE_PDF ($SOURCE_PDF) does not exist" >&2
  exit 2
fi

RERANKER_MODEL="${RERANKER_MODEL:-CompendiumLabs/bge-reranker-v2-m3-gguf}"
RERANKER_FILE="${RERANKER_FILE:-bge-reranker-v2-m3-Q4_K_M.gguf}"

SESSION="lilbee-rerank-qa"
SLICE_PDF="/tmp/lilbee-qa-slice.pdf"
LILBEE_DATA="$(mktemp -d -t lilbee-qa-XXXXXX)"
export LILBEE_DATA

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cleanup() {
  if [[ "${QA_KEEP:-0}" != "1" ]]; then
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    rm -rf "$LILBEE_DATA"
  else
    echo
    echo "QA_KEEP=1 - preserving state:"
    echo "  tmux session: $SESSION"
    echo "  LILBEE_DATA:  $LILBEE_DATA"
  fi
}
trap cleanup EXIT

command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 2; }
command -v uv >/dev/null || { echo "uv is required" >&2; exit 2; }

echo "[qa] slicing $SOURCE_PDF to first 5 pages -> $SLICE_PDF"
uv run python "$REPO_ROOT/scripts/_qa_slice_pdf.py" "$SOURCE_PDF" "$SLICE_PDF" 1-5

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 240 -y 60
tmux split-window -h -t "$SESSION"
tmux send-keys -t "${SESSION}.1" "tail -f $LILBEE_DATA/lilbee.log || true" C-m

send() { tmux send-keys -t "${SESSION}.0" "$1" C-m; }

echo "[qa] init + add"
send "cd $REPO_ROOT"
send "export LILBEE_DATA=$LILBEE_DATA"
send "uv run lilbee init"
send "uv run lilbee add $SLICE_PDF"

echo "[qa] Run A: reranker OFF"
send "unset LILBEE_RERANKER_MODEL"
send "uv run lilbee wiki build"
send "cp -r $LILBEE_DATA/wiki $LILBEE_DATA/../wiki-baseline"

echo "[qa] pulling reranker model $RERANKER_MODEL / $RERANKER_FILE"
send "uv run lilbee pull $RERANKER_MODEL --file $RERANKER_FILE || \
  hf download $RERANKER_MODEL $RERANKER_FILE \
  --local-dir $LILBEE_DATA/models/rerankers"

echo "[qa] Run B: reranker ON"
send "export LILBEE_RERANKER_MODEL=$RERANKER_FILE"
send "rm -rf $LILBEE_DATA/wiki"
send "uv run lilbee wiki build"
send "cp -r $LILBEE_DATA/wiki $LILBEE_DATA/../wiki-reranked"

echo "[qa] writing report"
REPORT="$LILBEE_DATA/../qa-report.md"
send "uv run python $REPO_ROOT/scripts/_qa_wiki_diff.py \
  $LILBEE_DATA/../wiki-baseline $LILBEE_DATA/../wiki-reranked \
  > $REPORT"

echo "[qa] done. tmux session: $SESSION"
echo "     report:             $REPORT"
echo "     baseline wiki:      $LILBEE_DATA/../wiki-baseline"
echo "     reranked wiki:      $LILBEE_DATA/../wiki-reranked"
echo
echo "Attach to inspect: tmux attach -t $SESSION"
echo "Keep state by rerunning with QA_KEEP=1."
