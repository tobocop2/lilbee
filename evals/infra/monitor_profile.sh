#!/usr/bin/env bash
# tmux monitor for a profile_ingest.sh run.
#
# The generic ingest monitor shows progress; this one shows the two things the
# profiling run exists to produce: the rate curve over time, and what worker 0
# is actually doing right now.
#
# Watchers are written to files and tmux runs the file: embedding them in the
# tmux command string loses panes silently, because they contain quotes.
set -uo pipefail
SESSION="${SESSION:-prof}"
tmux has-session -t "$SESSION" 2>/dev/null && exec tmux attach -t "$SESSION"
mkdir -p /root/mon

cat > /root/mon/rate.sh <<'EOF'
#!/usr/bin/env bash
# Rate between consecutive row-count samples. A falling column here IS the bug.
while :; do
  clear
  echo "=== throughput ==="
  # Lead with the running average. Writes flush 2000 chunks at a time, so a
  # per-sample rate is either "a batch landed" or 0 and reads as wild swings
  # that mean nothing; the average over the whole run is the real number.
  awk -F, 'NF==2 { if (!t0) {t0=$1; n0=$2} tN=$1; nN=$2 }
    END { if (tN>t0) printf "  RUNNING AVERAGE  %.1f docs/s   (%d rows in %ds)\n\n", (nN-n0)/(tN-t0), nN, tN-t0 }' \
    /root/prof/rows.csv 2>/dev/null
  echo "  per-sample deltas below are BURSTY BY DESIGN (2000-chunk flushes):"
  awk -F, 'NF==2 {
    if (prev_t) {
      dt = $1 - prev_t
      if (dt > 0) {
        rate = ($2 - prev_n) / dt
        if (!t0) t0 = prev_t
        printf "  %6ds  %10d rows  %8.1f docs/s\n", $1 - t0, $2, rate
      }
    }
    prev_t = $1; prev_n = $2
  }' /root/prof/rows.csv 2>/dev/null | tail -22
  sleep 20
done
EOF

cat > /root/mon/stack.sh <<'EOF'
#!/usr/bin/env bash
# What worker 0 is doing this instant. --idle would need the recorder; a dump is
# a point sample, so watch for which frames keep reappearing.
while :; do
  clear
  W0=$(pgrep -f "[l]ilbee sync" | head -1)
  echo "=== worker 0 (pid ${W0:-none}) stack right now ==="
  if [ -n "$W0" ]; then
    /root/venv/bin/py-spy dump --pid "$W0" 2>&1 | grep -vE "^Process |^Python v" | head -26
  else
    echo "  no worker running"
  fi
  sleep 10
done
EOF

cat > /root/mon/hot.sh <<'EOF'
#!/usr/bin/env bash
# Most frequent frames across every dump so far: the poor man's flame graph, and
# the thing that answers "where is the time going" while the run is still live.
while :; do
  clear
  echo "=== hottest frames across all samples so far ==="
  if [ -s /root/prof/stacks.txt ]; then
    grep -oE "^[[:space:]]+[a-zA-Z_][a-zA-Z0-9_]* \(" /root/prof/stacks.txt 2>/dev/null |
      sed 's/[[:space:]]*//; s/ ($//' | sort | uniq -c | sort -rn | head -20
    echo
    echo "  samples collected: $(grep -c '^=== ' /root/prof/stacks.txt 2>/dev/null)"
  else
    echo "  (no samples yet; py-spy attaches once worker 0 exists)"
  fi
  sleep 30
done
EOF

cat > /root/mon/gpu.sh <<'EOF'
#!/usr/bin/env bash
while :; do
  clear
  echo "=== GPUs (idle here + busy CPU elsewhere = starved pipeline) ==="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits |
    awk -F', ' '{printf "  card %s  util %3s%%  vram %6s MiB\n", $1, $2, $3}'
  echo
  echo "=== recent GPU samples (util per card) ==="
  tail -6 /root/prof/gpu.csv 2>/dev/null | cut -d, -f2- | sed 's/^/  /'
  sleep 5
done
EOF

cat > /root/mon/log.sh <<'EOF'
#!/usr/bin/env bash
while :; do
  clear
  echo "=== run log ==="
  grep -aE "^\[profile|^PROFILE|FATAL" /root/profile.log 2>/dev/null | tail -14
  echo
  echo "=== worker planning progress ==="
  for i in 0 1 2 3; do
    [ -f "/root/w$i/sync.log" ] || continue
    printf '  w%s %s\n' "$i" "$(tr '\r' '\n' < /root/w$i/sync.log | grep -a 'files/s' | tail -1 | cut -c1-90)"
  done
  sleep 15
done
EOF

for f in /root/mon/*.sh; do chmod +x "$f"; done

tmux new-session -d -s "$SESSION" -n profile -x 220 -y 55 /root/mon/rate.sh
tmux set -g pane-border-status top
tmux set -g pane-border-format ' #{pane_index} #{pane_title} '
tmux split-window -h -t "$SESSION:profile" /root/mon/hot.sh
tmux split-window -v -t "$SESSION:profile.0" /root/mon/log.sh
tmux split-window -v -t "$SESSION:profile.2" /root/mon/gpu.sh
tmux select-pane -t "$SESSION:profile.0" -T "rate over time"
tmux select-pane -t "$SESSION:profile.1" -T "run log"
tmux select-pane -t "$SESSION:profile.2" -T "hottest frames"
tmux select-pane -t "$SESSION:profile.3" -T "gpu"

tmux new-window -t "$SESSION:" -n stack /root/mon/stack.sh
tmux select-pane -t "$SESSION:stack.0" -T "worker 0 live stack"
tmux new-window -t "$SESSION:" -n shell
tmux select-window -t "$SESSION:profile"
exec tmux attach -t "$SESSION"
