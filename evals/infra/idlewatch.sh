#!/usr/bin/env bash
# Stop paying for GPUs the moment nobody is using them, without you watching.
#
# Ends the pod when either is true:
#   - the ingest and merge finished ($STATE_DIR/MERGE_DONE), after a grace
#     period, or
#   - nothing has been working for IDLE_MIN minutes, which is the crashed or
#     wedged case that never touches a marker.
#
# "Working" is deliberately several things, not one. Setup downloads a 1.3GB
# tarball, unpacks 8.8M files and pulls an 8GB model: roughly half an hour with
# no busy card and no worker process. A watchdog that only looked at GPUs and
# workers would power the box off in the middle of it, so the run script's own
# liveness counts, and so does a publish, which is network-bound and touches no
# card for its whole length.
#
# IT DELETES RATHER THAN STOPS, and that is a change from the previous run.
# Stopping looks safer and is not: it keeps the container disk but does NOT
# reserve the GPUs, so a stopped 8-pack could not be restarted after RunPod
# reallocated the cards, and the run on its disk was unrecoverable. The index now
# lives on a network volume that outlives the pod, so deleting is both cheaper
# (a stopped pod still bills for its disk) and strictly safer: the data is not on
# the thing being deleted. 'pod_native.sh resume' brings up a replacement pod on
# the same volume, and it is free to land on different hardware.
#
# Deleting needs the RunPod API, so the launcher drops a key at
# /root/.runpod/config.toml (mode 600). Without one this falls back to powering
# the box off, which is best effort inside a container; the pod's terminate-after
# is the hard backstop behind all of it.
set -uo pipefail
: "${STATE_DIR:=/workspace}"
: "${GRACE_MIN:=20}"     # after the run ends, how long to leave it up for a look
: "${IDLE_MIN:=30}"      # consecutive idle minutes that count as abandoned
: "${IDLE_UTIL:=5}"      # per-card utilisation below this is "not working"
LOG="$STATE_DIR/idlewatch.log"
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
  echo "$(( $(busy_cards) \
          + $(running "[l]ilbee sync") \
          + $(running "[n]ative9m.sh") \
          + $(running "[p]ublish9m.sh") ))"
}

end_pod() {
  local id; id=$(cat "$STATE_DIR/status/pod_id" 2>/dev/null || true)
  local ended=0
  if [ -n "$id" ] && command -v runpodctl >/dev/null 2>&1; then
    # The CLI has two shapes in the wild: 1.14, which is what a pod pulls from
    # the latest release, takes "remove pod <id>"; newer builds take "pod delete
    # <id>" and reject the old form. Trying only one left a measured pod RUNNING
    # after its run ended, which is the whole failure this watchdog prevents, so
    # try both rather than trusting one exit code.
    for verb in "pod delete" "remove pod"; do
      say "trying: runpodctl $verb $id"
      # shellcheck disable=SC2086
      if runpodctl $verb "$id" >>"$LOG" 2>&1; then
        ended=1; say "delete accepted via '$verb'"; break
      fi
    done
  else
    say "no runpodctl or pod id on this box"
  fi

  if [ "$ended" = "1" ]; then
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

say "armed: end ${GRACE_MIN}m after the merge, or after ${IDLE_MIN}m with no activity"
idle=0
while :; do
  if [ -f "$STATE_DIR/MERGE_DONE" ]; then
    say "merge finished; ${GRACE_MIN}m grace before ending the pod"
    # The grace period is the whole hold. An earlier version also waited on `who`
    # being empty, which is unusable here: the dashboard holds five pty-allocating
    # ssh sessions for the run's whole length, so `who` never empties and the pod
    # would bill until its terminate-after. Set GRACE_MIN for the look you want.
    sleep $(( GRACE_MIN * 60 ))
    break
  fi

  if [ "$(activity)" -eq 0 ]; then
    idle=$(( idle + 1 ))
    say "idle ${idle}/${IDLE_MIN} min (no busy card, no worker, no run script, no publish)"
    [ "$idle" -ge "$IDLE_MIN" ] && { say "abandoned: nothing has run for ${IDLE_MIN}m"; break; }
  else
    [ "$idle" -gt 0 ] && say "activity resumed"
    idle=0
  fi
  sleep 60
done

end_pod
