#!/usr/bin/env bash
# A tmux dashboard for the MS MARCO ingest.
#
# Six panes, each answering one question a long run raises: is it alive, are the
# GPUs actually working, how far along is it, what is it choking on, what is this
# costing. Read-only -- nothing here can disturb the run.
#
#   bash evals/infra/watch.sh   &&   tmux attach -t msmarco
set -euo pipefail

SESSION="${SESSION:-msmarco}"
CLUSTER="${CLUSTER:-msmarco2}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# ControlMaster keeps six panes from opening six SSH handshakes every refresh,
# which is what makes a dashboard like this stall on a busy pod.
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
     -o ControlMaster=auto -o ControlPath=/tmp/.ssh-msmarco-%r@%h:%p -o ControlPersist=300"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n launch -c "$REPO"
tmux send-keys -t "$SESSION:launch" "tail -f ${LAUNCH_LOG:-/tmp/msmarco2.log} | grep --line-buffered -viE 'Get:|Unpacking|Setting up|Preparing|Selecting|Reading database|warnings.warn|RequestsDependency'" C-m

# GPU utilisation. The number that says whether the box is embedding or waiting:
# a long run at 3% is a pipeline problem, not a slow model.
tmux new-window -t "$SESSION" -n gpu -c "$REPO"
tmux send-keys -t "$SESSION:gpu" \
  "watch -n 5 -t '$SSH $CLUSTER \"nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv\" 2>/dev/null'" C-m

# Progress: passages written to local disk, then rows in the index.
tmux new-window -t "$SESSION" -n progress -c "$REPO"
tmux send-keys -t "$SESSION:progress" \
  "watch -n 30 -t '$SSH $CLUSTER \"echo PASSAGES-ON-DISK:; find /root/msmarco/documents -type f -name \\\"*.txt\\\" 2>/dev/null | wc -l; echo; echo INDEX-SIZE:; du -sh /root/msmarco/data 2>/dev/null; echo; echo TRACED-DOCUMENTS:; wc -l < /workspace/logs/ingest_trace.log 2>/dev/null\" 2>/dev/null'" C-m

# The per-document trace: one line per extracted file, with elapsed_ms. This is
# what separates extraction cost from GPU cost after the run.
tmux new-window -t "$SESSION" -n trace -c "$REPO"
tmux send-keys -t "$SESSION:trace" \
  "$SSH $CLUSTER 'tail -f /workspace/logs/ingest_trace.log 2>/dev/null || echo \"trace not started yet\"'" C-m

# Anything that went wrong, rather than a count of things that went wrong.
tmux new-window -t "$SESSION" -n errors -c "$REPO"
tmux send-keys -t "$SESSION:errors" \
  "watch -n 60 -t '$SSH $CLUSTER \"grep -iE \\\"traceback|failed to extract|extraction failed|error\\\" /workspace/logs/*.log 2>/dev/null | tail -25 || echo no-errors-yet\" 2>/dev/null'" C-m

# What it is costing, and proof it is still alive.
tmux new-window -t "$SESSION" -n cost -c "$REPO"
tmux send-keys -t "$SESSION:cost" \
  "watch -n 60 -t 'cd $REPO && sky status 2>/dev/null | grep -E \"NAME|msmarco\"; echo; echo \"A100-80GB-SXM x1 = ~\\\$2.24/hr\"; echo; sky queue $CLUSTER 2>/dev/null | head -4'" C-m

# Watchdog. A silent process is the failure mode that cost this run two hours:
# sky hung twice with no output and no provision log, and both times it was
# noticed by a human rather than by anything watching.
tmux new-window -t "$SESSION" -n watchdog -c "$REPO"
tmux send-keys -t "$SESSION:watchdog" \
  "PREV=0; while true; do CUR=\$(wc -c < \${LAUNCH_LOG:-/tmp/msmarco2.log} 2>/dev/null || echo 0); \
   if [ \"\$CUR\" = \"\$PREV\" ]; then echo \"\$(date -u +%H:%M:%S) NO OUTPUT for 60s (size \$CUR)\"; \
   else echo \"\$(date -u +%H:%M:%S) advancing (\$CUR bytes)\"; fi; PREV=\$CUR; sleep 60; done" C-m

tmux select-window -t "$SESSION:launch"
echo "session '$SESSION' ready:  tmux attach -t $SESSION"
tmux list-windows -t "$SESSION" -F '  #{window_index}: #{window_name}'
