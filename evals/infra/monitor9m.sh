#!/usr/bin/env bash
# Build (or re-attach to) the tmux session that watches a 9M ingest on this pod.
#
# Windows: overview (4 panes), workers (one pane per card), gpu, merge, shell.
# Pane titles show in the border, so no pane is a mystery.
#
# Each watcher is written to /root/mon/<name>.sh and tmux runs the file. Embedding
# them in the tmux command string instead silently loses panes: the watchers
# contain single quotes, which terminate a 'bash -c' argument early, and a pane
# whose command fails to parse just disappears.
#
# Read-only: nothing here writes to the run's data roots. Row counts come from
# LanceDB once a minute, not every tick, because the writers hold a lock.
set -uo pipefail
SESSION="${SESSION:-ingest}"

tmux has-session -t "$SESSION" 2>/dev/null && exec tmux attach -t "$SESSION"

WORKERS=$(sed -n 's/^workers=//p' /root/status/run.env 2>/dev/null | head -1)
[ -n "${WORKERS:-}" ] || WORKERS=8
mkdir -p /root/mon

cat > /root/mon/progress.sh <<'EOF'
#!/usr/bin/env bash
VPY=$(cat /root/status/vpy 2>/dev/null || echo /root/venv/bin/python)
while :; do
  clear
  echo "=== progress ==="
  "$VPY" - <<'PY' 2>&1 | tail -20
import pathlib, time
st = pathlib.Path("/root/status")
if not (st / "run.env").exists():
    print("  (setup in progress: installing, or downloading the corpus)")
    raise SystemExit
env = {}
for line in (st / "run.env").read_text().splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
workers = int(env.get("workers", 8))
expected = int(env.get("expected", 8841823))
total = 0
try:
    import lancedb
except ImportError:
    print("  lancedb not importable in this interpreter")
    raise SystemExit
for i in range(workers):
    n = 0
    try:
        db = lancedb.connect(f"/root/w{i}/data/lancedb")
        listed = db.list_tables()
        names = list(getattr(listed, "tables", listed))
        if "_sources" in names:
            n = db.open_table("_sources").count_rows()
    except Exception:
        n = 0
    total += n
    print(f"  worker {i}: {n:>10,} rows")
print(f"  {'TOTAL':>9}: {total:>10,} / {expected:,}  ({100 * total / expected:.2f}%)")
started = st / "started_at"
if started.exists() and total:
    el = time.time() - int(started.read_text().strip())
    rate = total / el if el > 0 else 0
    left = (expected - total) / rate if rate > 0 else 0
    print(f"\n  elapsed {el / 3600:.2f}h    rate {rate:.1f} docs/s")
    print(f"  eta     {left / 3600:.2f}h    finish in {left / 60:.0f} min")
PY
  sleep 60
done
EOF

cat > /root/mon/plan.sh <<'EOF'
#!/usr/bin/env bash
# Planning is stat+hash in its own pool. Below ~61 files/s per worker it starves
# the cards, and nvidia-smi still reads 90%+ because the batches are undersized.
while :; do
  clear
  echo "=== planning rate (want > 61 files/s per worker) ==="
  for i in $(seq 0 7); do
    [ -f "/root/w$i/sync.log" ] || continue
    line=$(tr '\r' '\n' < "/root/w$i/sync.log" | grep -a 'files/s' | tail -1)
    printf '  w%s  %s\n' "$i" "$(printf '%s' "$line" | sed -n 's/.*examined \(.*\) left).*/\1/p')"
  done
  echo
  echo "=== exit codes (blank = running) ==="
  for i in $(seq 0 7); do printf '  w%s=%s' "$i" "$(cat "/root/w$i/rc" 2>/dev/null || printf -)"; done
  echo
  sleep 20
done
EOF

cat > /root/mon/gpu.sh <<'EOF'
#!/usr/bin/env bash
# A card at 0 MiB means its worker adopted a sibling's fleet instead of starting
# its own engine: that was the original all-but-one-card-idle failure.
while :; do
  clear
  echo "=== per-card (high util + low throughput = batch-starved) ==="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw \
             --format=csv,noheader,nounits |
    awk -F', ' '{printf "  card %s   util %3s%%   vram %6s MiB   %5s W\n", $1, $2, $3, $4}'
  sleep 5
done
EOF

cat > /root/mon/disk.sh <<'EOF'
#!/usr/bin/env bash
# The merge holds the shards and the merged copy at once: ~144GB of vectors each
# at 8.8M x 4096 dims, so running out mid-merge is the expensive failure.
while :; do
  clear
  echo "=== disk ==="
  df -h / | awk 'NR==1 || /\//{print "  " $0}'
  echo
  echo "=== shards ==="
  for i in $(seq 0 7); do
    [ -d "/root/w$i/data" ] || continue
    printf '  w%s  %s\n' "$i" "$(du -sh "/root/w$i/data" 2>/dev/null | cut -f1)"
  done
  [ -d /root/merged ] && printf '  merged  %s\n' "$(du -sh /root/merged 2>/dev/null | cut -f1)"
  echo
  echo "=== run ==="
  sed 's/^/  /' /root/status/run.env 2>/dev/null
  [ -f /root/RUN_DONE ] && echo "  RUN_DONE present"
  [ -f /root/FAILED_AT ] && echo "  *** FAILED, see the run-log pane ***"
  sleep 30
done
EOF

cat > /root/mon/files.sh <<'FILESEOF'
#!/usr/bin/env bash
# Files landing in the index right now, newest first, across all workers.
#
# Filtered server-side on ingested_at rather than pulling the table and sorting:
# at 1.1M rows a client-side sort every tick would cost more than the ingest.
# Rows land in write batches, so this arrives in bursts rather than a stream.
VPY=$(cat /root/status/vpy 2>/dev/null || echo /root/venv/bin/python)
while :; do
  clear
  echo "=== files just indexed (newest first, all workers) ==="
  "$VPY" /root/mon/recent_files.py 2>&1 | tail -28
  sleep 5
done
FILESEOF

cat > /root/mon/recent_files.py <<'PYEOF'
import datetime
import pathlib

import lancedb

run_env = pathlib.Path("/root/status/run.env")
if not run_env.exists():
    print("  (setup in progress: no workers started yet)")
    raise SystemExit
env = {}
for line in run_env.read_text().splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
workers = int(env.get("workers", 8))

def since(seconds):
    """Rows written in the last *seconds*, newest first, across every worker."""
    cut = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=seconds)).isoformat()
    out = []
    for i in range(workers):
        try:
            db = lancedb.connect(f"/root/w{i}/data/lancedb")
            rows = (
                db.open_table("_sources")
                .search()
                .where(f"ingested_at > '{cut}'")
                .select(["filename", "ingested_at", "chunk_count"])
                .limit(40)
                .to_list()
            )
            out += [(r["ingested_at"], i, r["filename"], r["chunk_count"]) for r in rows]
        except Exception:
            pass
    return out


# Writes land in batches and planning-heavy stretches emit nothing for minutes,
# so a fixed window shows an empty pane most of the time. Widen until there is
# something to show, and say how far back it had to look.
seen, window = [], 0
for window in (120, 900, 7200):
    seen = since(window)
    if seen:
        break

seen.sort(reverse=True)
if seen:
    print(f"  (most recent writes, looking back {window // 60} min)")
else:
    print("  (no rows yet: every worker is still planning its slice)")
for ts, worker, name, chunks in seen[:24]:
    print(f"  {ts[11:19]}  w{worker}  {name:<28} {chunks} chunk(s)")
PYEOF

cat > /root/mon/extract.sh <<'XEOF'
#!/usr/bin/env bash
# Live xberg extraction timings from LILBEE_INGEST_TRACE, every worker.
while :; do
  clear
  echo "=== xberg extraction latency, last 20k traced files (all workers) ==="
  cat /root/prof/w*.trace.log 2>/dev/null | tail -20000 |
    grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2 | sort -n |
    awk '{v[n++]=$1; s+=$1}
      END {
        if (!n) { print "  (no traces yet)"; exit }
        printf "  samples %d   mean %.2f ms\n", n, s/n
        printf "  p50 %s ms   p90 %s ms   p99 %s ms   max %s ms\n",
               v[int(n*0.50)], v[int(n*0.90)], v[int(n*0.99)], v[n-1]
      }'
  echo
  printf '  traced so far: %s files\n' "$(cat /root/prof/w*.trace.log 2>/dev/null | wc -l)"
  echo
  echo "=== most recent trace lines ==="
  tail -q -n 4 /root/prof/w*.trace.log 2>/dev/null | tail -8 | cut -c1-160
  sleep 15
done
XEOF

cat > /root/mon/throughput.sh <<'TEOF'
#!/usr/bin/env bash
# Writes flush 2000 chunks at a time, so a per-sample rate is either a batch or
# zero. The running average is the real number; buckets show the trend.
while :; do
  clear
  echo "=== throughput ==="
  awk -F, 'NF==2 { if (!t0) {t0=$1; n0=$2} tN=$1; nN=$2 }
    END { if (tN>t0) printf "  running average  %.1f docs/s   (%d rows in %.2fh)\n", \
          (nN-n0)/(tN-t0), nN, (tN-t0)/3600 }' /root/prof/rows.csv 2>/dev/null
  echo
  echo "  5-minute buckets:"
  awk -F, 'NF==2 {
      if (!t0) t0=$1
      b=int(($1-t0)/300); if (b>maxb) maxb=b
      if (pt) { d[b]+=$2-pn; s[b]+=$1-pt }
      pt=$1; pn=$2
    }
    END { for (i=0; i<=maxb; i++) if (s[i]>0) printf "    %4d-%4d min %8.1f docs/s\n", i*5, i*5+5, d[i]/s[i] }' \
    /root/prof/rows.csv 2>/dev/null | tail -13
  sleep 60
done
TEOF

cat > /root/mon/profile.sh <<'PEOF'
#!/usr/bin/env bash
while :; do
  clear
  echo "=== py-spy recorders ==="
  if pgrep -f "[p]y-spy record" >/dev/null; then
    pgrep -af "[p]y-spy record" | sed 's/^/  /' | cut -c1-150
  else
    echo "  none running (finished, or the preflight disabled profiling)"
  fi
  echo
  echo "=== recordings on disk ==="
  for f in /root/prof/*.folded /root/prof/*.svg; do
    [ -e "$f" ] || continue
    printf '  %-28s %s\n' "$(basename "$f")" "$(du -h "$f" 2>/dev/null | cut -f1)"
  done
  echo
  echo "=== host ==="
  tail -1 /root/prof/host.csv 2>/dev/null |
    awk -F, '{printf "  load %s   threads %s   rss %s MB\n", $2, $3, $4}'
  printf '  cores %s\n' "$(nproc)"
  sleep 20
done
PEOF

cat > /root/mon/runlog.sh <<'EOF'
#!/usr/bin/env bash
while :; do
  clear
  echo "=== run log ==="
  grep -aE '^\[ingest|^INGEST|^MERGE|^FATAL|merged ' /root/ingest.log 2>/dev/null | tail -22
  sleep 15
done
EOF

cat > /root/mon/merge.sh <<'EOF'
#!/usr/bin/env bash
echo "waiting for the merge phase (starts when every worker has finished)..."
tail -F /root/ingest.log 2>/dev/null |
  grep -aE --line-buffered 'merge|MERGE|merged |manifest|REFUSED'
EOF

for f in /root/mon/*.sh; do chmod +x "$f"; done

# ---- overview: four panes ---------------------------------------------------
tmux new-session -d -s "$SESSION" -n overview -x 200 -y 50 /root/mon/progress.sh
tmux set -g pane-border-status top
tmux set -g pane-border-format ' #{pane_index} #{pane_title} '
tmux split-window -h -t "$SESSION:overview" /root/mon/plan.sh
tmux split-window -v -t "$SESSION:overview.0" /root/mon/throughput.sh
tmux split-window -v -t "$SESSION:overview.2" /root/mon/disk.sh
tmux select-pane -t "$SESSION:overview.0" -T progress+eta
tmux select-pane -t "$SESSION:overview.1" -T throughput
tmux select-pane -t "$SESSION:overview.2" -T plan-rate
tmux select-pane -t "$SESSION:overview.3" -T disk+state

# ---- workers: one pane per card --------------------------------------------
tmux new-window -t "$SESSION:" -n workers "tail -F /root/w0/sync.log | tr '\r' '\n'"
tmux select-pane -t "$SESSION:workers.0" -T "w0 card0"
for i in $(seq 1 $((WORKERS - 1))); do
  tmux split-window -t "$SESSION:workers" "tail -F /root/w$i/sync.log | tr '\r' '\n'"
  tmux select-pane -t "$SESSION:workers" -T "w$i card$i"
  tmux select-layout -t "$SESSION:workers" tiled >/dev/null
done

# ---- files: what is landing right now --------------------------------------
tmux new-window -t "$SESSION:" -n files /root/mon/files.sh
tmux select-pane -t "$SESSION:files.0" -T "indexed just now"
tmux split-window -h -t "$SESSION:files" /root/mon/runlog.sh
tmux select-pane -t "$SESSION:files.1" -T "run log"

tmux new-window -t "$SESSION:" -n extract /root/mon/extract.sh
tmux select-pane -t "$SESSION:extract.0" -T "xberg extraction latency"

tmux new-window -t "$SESSION:" -n profile /root/mon/profile.sh
tmux select-pane -t "$SESSION:profile.0" -T "py-spy + host"

# ---- gpu -------------------------------------------------------------------
tmux new-window -t "$SESSION:" -n gpu /root/mon/gpu.sh
tmux select-pane -t "$SESSION:gpu.0" -T per-card
tmux split-window -h -t "$SESSION:gpu" "nvidia-smi dmon -s um"
tmux select-pane -t "$SESSION:gpu.1" -T dmon

# ---- merge -----------------------------------------------------------------
tmux new-window -t "$SESSION:" -n merge /root/mon/merge.sh
tmux select-pane -t "$SESSION:merge.0" -T merge+verify

# ---- shell -----------------------------------------------------------------
tmux new-window -t "$SESSION:" -n shell
tmux select-pane -t "$SESSION:shell.0" -T scratch

tmux select-window -t "$SESSION:overview"
exec tmux attach -t "$SESSION"
