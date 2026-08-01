#!/usr/bin/env bash
# Delete the GPU pod before the account reaches zero, keeping the volume.
#
# A pod that runs the balance to zero does not just stop: RunPod reclaims
# unpaid resources, and the network volume holding the index is one of them.
# Ending the pod with a few dollars left keeps the volume alive and paid for at
# roughly $0.06/hr, so the ingest can resume the moment the account is topped up.
#
# This is the local counterpart of idlewatch.sh: that one watches for a finished
# or wedged run from the pod, this one watches the account from the laptop, which
# is the only place that can see a balance the pod is about to exhaust.
#
#   balance_guard.sh start   arm it (backgrounds itself)
#   balance_guard.sh stop
#   balance_guard.sh status
set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
: "${FLOOR:=3.00}"        # dollars below which the pod goes
: "${EVERY:=120}"         # seconds between checks
STATE="$HOME/.msmarco9m/run.json"
GUARD_DIR="$HOME/.msmarco9m"
PIDFILE="$GUARD_DIR/.guard.pid"
LOG="$GUARD_DIR/guard.log"

say() { printf '[guard %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >> "$LOG"; }

balance() {
  runpodctl user 2>/dev/null \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['clientBalance'])" 2>/dev/null \
    || echo 999
}

loop() {
  say "armed: pod goes below \$$FLOOR so the volume stays paid for"
  while :; do
    [ -f "$PIDFILE" ] || { say "pidfile gone; standing down"; exit 0; }
    local bal pod
    bal=$(balance)
    pod=$(python3 -c "
import json, pathlib
p = pathlib.Path('$STATE')
print(json.loads(p.read_text()).get('pod', '') if p.exists() else '')" 2>/dev/null)
    [ -n "$pod" ] || { say "no pod recorded; standing down"; rm -f "$PIDFILE"; exit 0; }
    if awk -v b="$bal" -v f="$FLOOR" 'BEGIN{exit !(b < f)}'; then
      say "balance \$$bal is below \$$FLOOR: deleting pod $pod, KEEPING the volume"
      runpodctl pod delete "$pod" >>"$LOG" 2>&1
      python3 -c "
import json, pathlib
p = pathlib.Path('$STATE')
d = json.loads(p.read_text()); d['pod'] = ''; d['stopped_for_balance'] = True
p.write_text(json.dumps(d, indent=1))" 2>/dev/null
      say "pod deleted. 'pod_native.sh resume' continues once the account is funded."
      rm -f "$PIDFILE"
      exit 0
    fi
    sleep "$EVERY"
  done
}

case "${1:-status}" in
  start)
    mkdir -p "$GUARD_DIR"
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "already armed (pid $(cat "$PIDFILE"))"; exit 0
    fi
    nohup "$SELF" _loop >/dev/null 2>&1 &
    echo $! > "$PIDFILE"
    echo "balance guard armed at \$$FLOOR (pid $(cat "$PIDFILE")); log at $LOG"
    ;;
  _loop) loop ;;
  stop) rm -f "$PIDFILE"; pkill -f "balance_guard.sh _loop" 2>/dev/null; echo "disarmed" ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "armed (pid $(cat "$PIDFILE")), floor \$$FLOOR, balance \$$(balance)"
    else
      echo "not armed; balance \$$(balance)"
    fi
    tail -5 "$LOG" 2>/dev/null | sed 's/^/  /'
    ;;
  *) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
