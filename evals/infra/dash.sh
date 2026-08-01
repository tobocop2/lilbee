#!/usr/bin/env bash
# Live dashboard for a full-corpus ingest, in two places from one layout.
#
#   dash.sh up       build the session on THIS machine; each pane is an ssh that
#                    reads the pod. Your tmux.conf applies, and this is what the
#                    recorder captures.
#   dash.sh attach   re-attach locally, any time, without disturbing a recording
#   dash.sh serve    build the same session ON THE POD, for when you are ssh'd in
#   dash.sh pane <n> one pane's renderer; runs wherever the data is
#
# WHY BOTH. The recording has to be local, because the reel should carry your
# tmux configuration and cost the box nothing. But a dashboard you can only see
# through the recorder is a dashboard you cannot check from a phone over ssh, so
# the same five panes are also servable from the pod. One renderer, two hosts.
#
# ATTACHING DOES NOT DISTURB THE RECORDING. The session's geometry is pinned
# (window-size manual, 210x54), so a second client cannot resize it, and the
# recorder attaches read-only. Attach, detach, resize your terminal: the cast
# keeps its shape.
#
# PANE SIZES ARE CHOSEN BY CONTENT, not by tmux's default halving. The extraction
# trace lines are 118 columns, so that pane gets the full width rather than
# wrapping every line three times; the ingest pane needs 8 worker rows plus a
# total plus lilbee's own bar, which is 16 rows.
set -uo pipefail

: "${SESSION:=ingest}"
: "${VOL:=/workspace}"
: "${COLS:=210}"
: "${ROWS:=54}"
STATE="$HOME/.msmarco9m/run.json"
# The corpus size is read from the run's own counted total rather than passed in:
# a trial and the full run then need no different invocation, and the dashboard
# cannot disagree with the guard about how many files there are.
: "${EXPECTED:=$(cat "$VOL/status/expected" 2>/dev/null || echo 0)}"

C_HDR=$'\033[1;36m'; C_OK=$'\033[1;32m'; C_WARN=$'\033[1;33m'; C_DIM=$'\033[2;37m'
C_NUM=$'\033[1;35m'; R=$'\033[0m'

commas() { awk -v n="${1:-0}" 'BEGIN{s=sprintf("%d",n); r="";
  while (length(s) > 3) { r = "," substr(s, length(s)-2) r; s = substr(s, 1, length(s)-3) }
  print s r }'; }

bar() {  # $1 = filled cells, $2 = width
  local n=${1:-0} w=${2:-30} i
  [ "$n" -gt "$w" ] && n=$w
  printf '%s' "$C_OK"; for ((i = 0; i < n; i++)); do printf '▰'; done
  printf '%s' "$C_DIM"; for ((i = n; i < w; i++)); do printf '▱'; done
  printf '%s' "$R"
}

# --- panes --------------------------------------------------------------------

pane_ingest() {
  while :; do
    clear
    printf '%s  lilbee sync%s  %sone worker per card, native fan-out%s\n\n' \
      "$C_HDR" "$R" "$C_DIM" "$R"
    # lilbee's own aggregate bar, verbatim: it is the product's report of its own
    # progress and nothing here should paraphrase it.
    tail -c 4000 "$VOL/sync.out" 2>/dev/null | tr '\r' '\n' | grep -a '[^[:space:]]' \
      | tail -2 | sed 's/^/   /'
    echo
    # Per-worker rows come from each shard's own store: the parent aggregates the
    # counters in memory and reports one bar, so per-worker progress is only
    # observable from the shards themselves.
    local line ts merged per total pct elapsed rate eta started
    line=$(tail -1 "$VOL/prof/rows.csv" 2>/dev/null)
    if [ -z "$line" ]; then
      printf '   %s(waiting for the first worker to write a row)%s\n' "$C_DIM" "$R"
    else
      ts=${line%%,*}; line=${line#*,}
      merged=${line%%,*}; per=${line#*,}
      [ "$per" = "$merged" ] && per=""
      total=0
      local i=0 n
      for n in ${per//,/ }; do
        total=$((total + n))
        printf '   %sworker%s %s%s%s  %s  %s%12s%s\n' \
          "$C_DIM" "$R" "$C_NUM" "$i" "$R" \
          "$(bar $(( EXPECTED > 0 ? n * 8 * 40 / EXPECTED : 0 )) 40)" \
          "$C_NUM" "$(commas "$n")" "$R"
        i=$((i + 1))
      done
      # After the merge the shards stop moving and the merged index is the truth.
      [ "${merged:-0}" -gt "$total" ] && total=$merged
      # Ternaries are parenthesised: an unbracketed `>` inside printf's argument
      # list parses as an output redirection, and awk then fails the whole
      # statement rather than the comparison, so docs/s and the ETA vanish.
      pct=$(awk -v a="$total" -v b="$EXPECTED" 'BEGIN{printf "%.2f", (b ? a * 100 / b : 0)}')
      # Rate over a TRAILING WINDOW, not since the sync started. The first six
      # minutes of a run produce no rows at all (every worker walks the whole
      # corpus, then eight llama-servers load an 8GB model each), so dividing by
      # total elapsed reports about two thirds of the real rate and an ETA half
      # again too long. The window is the last 30 samples, which is ten minutes.
      read -r rate eta <<<"$(tail -30 "$VOL/prof/rows.csv" 2>/dev/null | awk -F, -v b="$EXPECTED" '
        { m = $2 + 0; s = 0; for (i = 3; i <= NF; i++) s += $i; n = (m > s ? m : s)
          if (!t0) { t0 = $1; n0 = n }
          tN = $1; nN = n }
        END { d = tN - t0
              r = (d > 0 ? (nN - n0) / d : 0)
              printf "%.1f %.1f", r, (r > 0 ? (b - nN) / r / 3600 : 0) }')"
      printf '\n   %sindexed%s %s%s%s %s/ %s%s   %s%s%%%s   %s%s docs/s%s   %seta %sh%s\n' \
        "$C_HDR" "$R" "$C_NUM" "$(commas "$total")" "$R" \
        "$C_DIM" "$(commas "$EXPECTED")" "$R" "$C_OK" "$pct" "$R" \
        "$C_NUM" "$rate" "$R" "$C_DIM" "$eta" "$R"
      printf '   %s%s%s\n' "$C_DIM" "$(bar $((EXPECTED > 0 ? total * 60 / EXPECTED : 0)) 60)" "$R"
    fi
    sleep 3
  done
}

pane_extract() {
  while :; do
    clear
    printf '%s  xberg extraction%s  %severy file, exact percentiles%s\n\n' \
      "$C_HDR" "$R" "$C_DIM" "$R"
    # Percentiles over the WHOLE corpus, from the incremental histogram the run
    # keeps current; a tail sample would describe the last few minutes only.
    if [ -s "$VOL/prof/extract.summary" ]; then
      sed "s/^/  /" "$VOL/prof/extract.summary"
    else
      printf '   %s(no traces yet)%s\n' "$C_DIM" "$R"
    fi
    echo
    # Full width, with the logging prefix dropped: the timestamp and logger name
    # are 60 columns of the 190 and say nothing the pane header does not.
    tail -8 "$VOL/prof/extract.trace.log" 2>/dev/null \
      | sed "s/^.*trace: //; s|/root/corpus/documents/||" \
      | sed "s/^/   /; s/elapsed_ms=/${C_DIM}elapsed_ms=${R}${C_OK}/; s/ chunks=/${R} ${C_DIM}chunks=/" \
      | cut -c1-206
    sleep 5
  done
}

pane_gpu() {
  while :; do
    clear
    printf '%s  GPUs%s\n\n' "$C_HDR" "$R"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', ' -v ok="$C_OK" -v dim="$C_DIM" -v num="$C_NUM" -v warn="$C_WARN" -v r="$R" \
        '{printf "   %scard%s %s%s%s  %s%3s%%%s  %s%6s MiB%s  %s%4s W%s\n",
                 dim, r, num, $1, r, ($2 > 50 ? ok : warn), $2, r, dim, $3, r, warn, int($4), r}'
    # p10 and the share of readings at exactly zero, because the mean hides
    # starvation: a fed fleet and a starved one both read ~90% while busy.
    # Sorted by sort(1) rather than in awk: mawk is the default here and has no
    # asort, so an awk-side sort works when developed and not on the box.
    echo
    cut -d, -f2- "$VOL/prof/gpu.csv" 2>/dev/null | tr ',' '\n' | grep -E '^[0-9]+$' \
      | sort -n \
      | awk -v dim="$C_DIM" -v r="$R" '{ v[n++] = $1; if ($1 == 0) z++ }
        END { if (!n) { printf "   %s(no samples yet)%s\n", dim, r; exit }
              printf "   %sp10%s %d%%   %sp50%s %d%%   %sat zero%s %.1f%%   %sn%s %d\n",
                dim, r, v[int(n * 0.1)], dim, r, v[int(n * 0.5)],
                dim, r, 100 * z / n, dim, r, n }'
    sleep 4
  done
}

pane_host() {
  while :; do
    clear
    printf '%s  host%s\n\n' "$C_HDR" "$R"
    local line load threads mem
    line=$(tail -1 "$VOL/prof/host.csv" 2>/dev/null)
    load=$(echo "$line" | cut -d, -f2); threads=$(echo "$line" | cut -d, -f3)
    mem=$(echo "$line" | cut -d, -f4)
    printf '   %sload%s     %s%s%s %sof %s cores%s\n' "$C_DIM" "$R" "$C_NUM" \
      "${load:-?}" "$R" "$C_DIM" "$(nproc 2>/dev/null || echo '?')" "$R"
    printf '   %sthreads%s  %s%s%s\n' "$C_DIM" "$R" "$C_NUM" "${threads:-?}" "$R"
    printf '   %smem free%s %s%s MB%s\n' "$C_DIM" "$R" "$C_NUM" "${mem:-?}" "$R"
    printf '   %swatts%s    %s%s%s\n' "$C_DIM" "$R" "$C_WARN" \
      "$(tail -1 "$VOL/prof/sys.csv" 2>/dev/null | cut -d, -f2)" "$R"
    echo
    printf '   %sphase%s\n   %s%s%s\n' "$C_DIM" "$R" "$C_OK" \
      "$(cat "$VOL/status/phase" 2>/dev/null || echo '?')" "$R"
    sleep 5
  done
}

pane_disk() {
  while :; do
    clear
    printf '%s  volume + state%s\n\n' "$C_HDR" "$R"
    # `du`, not `df`: the volume is a MooseFS mount, so df reports the whole
    # cluster (630T of 851T) rather than anything about this run.
    printf '   %sused%s    %s%s%s\n' "$C_DIM" "$R" "$C_NUM" \
      "$(du -sh "$VOL" 2>/dev/null | cut -f1)" "$R"
    printf '   %sindex%s   %s%s%s\n' "$C_DIM" "$R" "$C_NUM" \
      "$(du -sh "$VOL/kb/.lilbee/data" 2>/dev/null | cut -f1)" "$R"
    printf '   %sshards%s  %s%s%s\n' "$C_DIM" "$R" "$C_NUM" \
      "$(du -sh "$VOL/kb/.lilbee/shards" 2>/dev/null | cut -f1)" "$R"
    printf '   %straces%s  %s%s%s\n\n' "$C_DIM" "$R" "$C_NUM" \
      "$(du -sh "$VOL/prof" 2>/dev/null | cut -f1)" "$R"
    local marker
    for marker in MERGE_DONE PUBLISH_DONE COUNT_MISMATCH FAILED_AT; do
      [ -e "$VOL/$marker" ] && printf '   %s%s%s\n' "$C_WARN" "$marker" "$R"
    done
    if pgrep -f '[l]ilbee sync' >/dev/null 2>&1; then
      printf '   %s● running%s  %s%s workers%s\n' "$C_OK" "$R" "$C_DIM" \
        "$(pgrep -fc '[l]ilbee' 2>/dev/null | head -1)" "$R"
    else
      printf '   %s○ no sync process%s\n' "$C_DIM" "$R"
    fi
    sleep 6
  done
}

# --- layout -------------------------------------------------------------------

# Same five panes wherever they run; only how the command reaches the data differs.
build() {  # $1 = command template; @PANE@ is replaced with each pane's name
  local run="$1" main mid bl bm br
  tmux new-session -d -s "$SESSION" -n ingest -x "$COLS" -y "$ROWS"
  # Pinned geometry, so a second client (yours) cannot reshape what the recorder
  # is capturing, and the cast keeps one size end to end.
  tmux set-option -t "$SESSION" window-size manual
  tmux resize-window -t "$SESSION:ingest" -x "$COLS" -y "$ROWS" 2>/dev/null

  # Captured pane ids, never indices: pane-base-index is 1 in this tmux.conf and
  # a hardcoded .0 addresses the wrong pane.
  main=$(tmux list-panes -t "$SESSION:ingest" -F '#{pane_id}' | head -1)
  mid=$(tmux split-window -v -P -F '#{pane_id}' -t "$main")
  bl=$(tmux split-window -v -P -F '#{pane_id}' -t "$mid")
  bm=$(tmux split-window -h -P -F '#{pane_id}' -t "$bl")
  br=$(tmux split-window -h -P -F '#{pane_id}' -t "$bm")

  tmux send-keys -t "$main" "${run//@PANE@/ingest}" C-m
  tmux send-keys -t "$mid"  "${run//@PANE@/extract}" C-m
  tmux send-keys -t "$bl"   "${run//@PANE@/gpu}" C-m
  tmux send-keys -t "$bm"   "${run//@PANE@/host}" C-m
  tmux send-keys -t "$br"   "${run//@PANE@/disk}" C-m

  tmux select-pane -t "$main" -T 'ingest · per worker · indexed'
  tmux select-pane -t "$mid"  -T 'xberg extraction'
  tmux select-pane -t "$bl"   -T 'gpu'
  tmux select-pane -t "$bm"   -T 'host'
  tmux select-pane -t "$br"   -T 'volume'

  # Content-driven heights, and -y counts the pane's border row, so each is one
  # more than the content needs. Ingest: header, lilbee's two lines, 8 worker
  # bars, the total and its bar = 16 rows of content. Extraction: header, three
  # lines of percentiles and 8 trace lines = 14. Undersizing these does not
  # truncate, it SCROLLS, and a scrolled pane shows only its last line.
  tmux resize-pane -t "$main" -y 18
  tmux resize-pane -t "$mid" -y 16
  tmux resize-pane -t "$bl" -x 104
  tmux resize-pane -t "$bm" -x 52

  tmux set -t "$SESSION" pane-border-status top
  tmux set -t "$SESSION" pane-border-format ' #[bold]#{pane_title} '
  tmux select-pane -t "$main"
}

remote_prefix() {
  [ -f "$STATE" ] || { echo "no run recorded at $STATE" >&2; exit 1; }
  local host port
  host=$(python3 -c "import json;print(json.load(open('$STATE')).get('host',''))")
  port=$(python3 -c "import json;print(json.load(open('$STATE')).get('port',''))")
  [ -n "$host" ] && [ -n "$port" ] || { echo "state has no endpoint" >&2; exit 1; }
  # Each pane reconnects on its own, so a dropped ssh leaves a retrying pane
  # rather than a dead one in the middle of a six-hour recording.
  #
  # One connection PER PANE, deliberately. Multiplexing them through a shared
  # ControlMaster looks tidier and is worse: every client then dies together
  # whenever the master drops, and five panes losing their data at once reads
  # exactly like a dead pod. Five long-lived authenticated sessions are well
  # inside sshd's limits; the apparent saturation that prompted the change was
  # the shared socket failing, not sshd refusing.
  printf 'while :; do ssh -tt -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
  printf -- '-o ConnectTimeout=15 -o IdentitiesOnly=yes -o ServerAliveInterval=30 '
  printf -- '-o ControlMaster=no -o ControlPath=none '
  printf -- '-i %s/.ssh/runpod_qa -p %s root@%s ' "$HOME" "$port" "$host"
  printf -- '"bash /root/dash.sh pane @PANE@" || echo "  [reconnecting]"; sleep 5; done'
}

case "${1:-attach}" in
  pane)
    case "${2:-ingest}" in
      ingest) pane_ingest ;; extract) pane_extract ;; gpu) pane_gpu ;;
      host) pane_host ;; disk) pane_disk ;;
      *) echo "unknown pane: ${2:-}" >&2; exit 1 ;;
    esac
    ;;
  start)
    # Build it and leave it running detached, so a driver can start the session
    # and the recorder without needing a terminal to attach to.
    tmux has-session -t "$SESSION" 2>/dev/null || build "$(remote_prefix)"
    echo "session '$SESSION' ready (${COLS}x${ROWS}); attach with: dash.sh attach"
    ;;
  up)
    tmux has-session -t "$SESSION" 2>/dev/null || build "$(remote_prefix)"
    exec tmux attach -t "$SESSION"
    ;;
  serve)
    # On the pod: the same layout, reading the volume directly. Addressed through
    # $0 and carrying VOL, so this also runs against a staged volume anywhere.
    SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    tmux has-session -t "$SESSION" 2>/dev/null \
      || build "VOL=$VOL EXPECTED=$EXPECTED bash $SELF pane @PANE@"
    exec tmux attach -t "$SESSION"
    ;;
  attach)
    tmux has-session -t "$SESSION" 2>/dev/null || { echo "no session; run 'dash.sh up'" >&2; exit 1; }
    exec tmux attach -t "$SESSION"
    ;;
  kill) tmux kill-session -t "$SESSION" 2>/dev/null; echo "killed $SESSION" ;;
  *) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
