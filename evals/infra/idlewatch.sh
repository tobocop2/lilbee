#!/usr/bin/env bash
# Stop paying for GPUs the moment nobody is using them, without you watching.
#
# Stops the pod when either is true:
#   - the run finished or failed (/root/RUN_DONE), after a grace period, or
#   - nothing has been working for IDLE_MIN minutes, which is the crashed or
#     wedged case that never touches RUN_DONE.
#
# "Working" is deliberately three things, not one. The setup phase downloads a
# 1.3GB tarball, unpacks 8.8M files, hard-links them into per-worker trees and
# pulls an 8GB model: half an hour with no busy card and no worker process. A
# watchdog that only looked at GPUs and workers would power the box off in the
# middle of it, so the ingest script's own liveness counts as activity too.
#
# It STOPS rather than deletes: GPU billing ends, the container disk and every
# result survive, and 'pod9m.sh resume' brings it back to fetch them. Stopping
# needs the RunPod API, so the launcher drops a key at /root/.runpod/config.toml
# (mode 600) unless NO_SELF_STOP=1, in which case this falls back to powering the
# box off, which is best-effort inside a container. The pod's terminate-after is
# the hard backstop behind all of it.
set -uo pipefail
: "${GRACE_MIN:=20}"     # after the run ends, how long to leave it up for a look
: "${IDLE_MIN:=30}"      # consecutive idle minutes that count as abandoned
: "${IDLE_UTIL:=5}"      # per-card utilisation below this is "not working"
LOG=/root/idlewatch.log
say() { printf '[idlewatch %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >> "$LOG"; }

busy_cards() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null |
    awk -v t="$IDLE_UTIL" '$1 > t {n++} END {print n+0}'
}
# Bracketed so the pattern cannot match this watchdog's own command line.
# pgrep -c prints 0 AND exits non-zero when nothing matches, so a `|| echo 0`
# fallback emits two lines and every later sum becomes a syntax error: the
# watchdog then never fires, which is the failure that costs a night of GPU.
running() {
  local n
  n=$(pgrep -fc "$1" 2>/dev/null | head -1)
  echo "${n:-0}"
}
activity() {
  local cards workers driver
  cards=$(busy_cards)
  workers=$(running "[l]ilbee sync")
  driver=$(running "[i]ngest9m.sh")
  echo "$(( cards + workers + driver ))"
}

stop_pod() {
  local id; id=$(cat /root/status/pod_id 2>/dev/null || true)
  local stopped=0
  if [ -n "$id" ] && command -v runpodctl >/dev/null 2>&1; then
    # The CLI has two shapes in the wild: 1.14, which is what the pod pulls from
    # the latest release, takes "stop pod <id>"; newer builds take "pod stop
    # <id>" and reject the old form. Trying only one left a measured pod RUNNING
    # after its run ended, which is the whole failure this watchdog exists to
    # prevent, so try both and verify rather than trusting an exit code.
    for verb in "stop pod" "pod stop"; do
      say "trying: runpodctl $verb $id"
      # shellcheck disable=SC2086
      if runpodctl $verb "$id" >>"$LOG" 2>&1; then
        stopped=1; say "stop accepted via '$verb'"; break
      fi
    done
  else
    say "no runpodctl or pod id on this box"
  fi

  if [ "$stopped" = "1" ]; then
    say "waiting for the container to go away"
    sleep 180
  fi
  # poweroff inside a container returns success and does nothing, which
  # short-circuits an || chain, so killing init is the terminal action rather
  # than the last link of one.
  say "stopping the container directly"
  sync
  poweroff 2>/dev/null
  sleep 10
  kill -9 1 2>/dev/null
}

say "armed: stop ${GRACE_MIN}m after the run ends, or after ${IDLE_MIN}m with no activity"
idle=0
while :; do
  if [ -f /root/RUN_DONE ]; then
    say "run finished; ${GRACE_MIN}m grace before stopping"
    sleep $(( GRACE_MIN * 60 ))
    # Someone poking around inside the grace window keeps it up until they leave.
    if who 2>/dev/null | grep -q .; then
      say "an ssh session is open; holding until it ends"
      while who 2>/dev/null | grep -q .; do sleep 300; done
      say "sessions ended"
    fi
    break
  fi

  if [ "$(activity)" -eq 0 ]; then
    idle=$(( idle + 1 ))
    say "idle ${idle}/${IDLE_MIN} min (no busy card, no worker, no ingest script)"
    [ "$idle" -ge "$IDLE_MIN" ] && { say "abandoned: nothing has run for ${IDLE_MIN}m"; break; }
  else
    [ "$idle" -gt 0 ] && say "activity resumed"
    idle=0
  fi
  sleep 60
done

stop_pod
