#!/usr/bin/env bash
# Record the local dashboard for the whole run, in hourly segments.
#
#   rec.sh start   begin recording (backgrounds itself; survives your terminal)
#   rec.sh stop    stop after the current segment
#   rec.sh status  segments so far, sizes, whether it is running
#   rec.sh render  turn a segment into a GIF
#
# WHY SEGMENTS. A single six-hour cast is one file that a crash at hour five
# loses entirely, and asciinema only writes its header on a clean exit. An hour
# is short enough that a failure costs one segment and long enough that the
# seams are rare. Each finished segment uploads immediately, so the recording is
# safe as it is made rather than at the end.
#
# WHY IT COSTS THE POD NOTHING. The recorder attaches to the LOCAL tmux session,
# which is already reading the pod over ssh for the dashboard. Recording adds no
# process, no sample and no byte on the box. The attach is read-only, so the
# recorder cannot type into the session, and the session's geometry is pinned, so
# your own attach cannot reshape what is being recorded.
set -uo pipefail
# Absolute, because the segment loop is re-entered through nohup: a bare "$0"
# there is resolved against PATH, not the working directory, and a relative
# invocation silently records nothing.
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
: "${SESSION:=ingest}"
: "${REC_DIR:=$HOME/msmarco-ingest/rec}"
: "${SEGMENT_S:=3600}"
: "${COLS:=210}"
: "${ROWS:=54}"
: "${IDLE_LIMIT:=2}"          # collapse dead air; a dashboard is mostly still
: "${HF_REPO:=beeberg/msmarco-ingest-checkpoint}"
: "${UPLOAD:=1}"
PIDFILE="$REC_DIR/.rec.pid"
STOPFILE="$REC_DIR/.stop"

log() { printf '[rec %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

need() { command -v "$1" >/dev/null || { echo "error: $1 not on PATH" >&2; exit 1; }; }

upload() {  # $1 = finished cast
  [ "$UPLOAD" = "1" ] || return 0
  python3 - "$1" "$HF_REPO" <<'PY' 2>&1 | sed 's/^/  /'
import os
import pathlib
import sys

from huggingface_hub import HfApi

cast, repo = pathlib.Path(sys.argv[1]), sys.argv[2]
token = (os.environ.get("HF_TOKEN")
         or pathlib.Path.home().joinpath(".cache/huggingface/token").read_text().strip())
api = HfApi(token=token)
api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_file(
    path_or_fileobj=str(cast), path_in_repo=f"recording/{cast.name}",
    repo_id=repo, repo_type="dataset", commit_message=f"recording segment {cast.stem}",
)
print(f"uploaded recording/{cast.name}")
PY
}

loop() {
  local n=0 cast
  while [ ! -f "$STOPFILE" ]; do
    tmux has-session -t "$SESSION" 2>/dev/null || { log "no session $SESSION; waiting"; sleep 20; continue; }
    n=$((n + 1))
    cast=$(printf '%s/ingest-%03d.cast' "$REC_DIR" "$n")
    [ -e "$cast" ] && continue   # resumed run: skip past segments already recorded
    log "segment $n -> $cast"
    # Read-only attach, so the recorder is a spectator: it cannot send keys into
    # the session and cannot change its size.
    # SIGTERM at the segment boundary; asciinema finalises the file on it.
    timeout -s TERM "$SEGMENT_S" \
      asciinema rec --headless --window-size "${COLS}x${ROWS}" -i "$IDLE_LIMIT" \
        --overwrite -c "tmux attach -r -t $SESSION" "$cast" >/dev/null 2>&1
    if [ -s "$cast" ]; then
      log "segment $n: $(du -h "$cast" | cut -f1)"
      upload "$cast"
    else
      log "segment $n produced nothing; the session is probably gone"
      rm -f "$cast"
      sleep 20
    fi
  done
  log "stopped after $n segment(s)"
  rm -f "$PIDFILE" "$STOPFILE"
}

case "${1:-status}" in
  start)
    need asciinema; need tmux; need timeout
    mkdir -p "$REC_DIR"
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "already recording (pid $(cat "$PIDFILE"))"; exit 0
    fi
    rm -f "$STOPFILE"
    nohup "$SELF" _loop >> "$REC_DIR/rec.log" 2>&1 &
    echo $! > "$PIDFILE"
    echo "recording to $REC_DIR (pid $(cat "$PIDFILE")); log at $REC_DIR/rec.log"
    ;;
  _loop) loop ;;
  stop)
    touch "$STOPFILE"
    [ -f "$PIDFILE" ] && pkill -P "$(cat "$PIDFILE")" -f asciinema 2>/dev/null
    echo "will stop after the current segment finalises"
    ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "recording (pid $(cat "$PIDFILE"))"
    else
      echo "not recording"
    fi
    ls -lh "$REC_DIR"/*.cast 2>/dev/null | awk '{printf "  %s  %s\n", $9, $5}' || echo "  (no segments)"
    ;;
  render)
    need agg
    cast="${2:-$(ls "$REC_DIR"/*.cast 2>/dev/null | head -1)}"
    [ -f "$cast" ] || { echo "no cast to render" >&2; exit 1; }
    out="${cast%.cast}.gif"
    agg --font-family "FiraCode Nerd Font" --theme kanagawa --speed "${SPEED:-8}" \
      "$cast" "$out" || exit 1
    echo "$out ($(du -h "$out" | cut -f1))"
    ;;
  *) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
