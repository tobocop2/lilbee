#!/usr/bin/env bash
# Open a local tmux session that SSHes to the running pod and shows the monitor.
#
# Local session name is fixed so re-running attaches instead of stacking sessions,
# and so it never touches any other tmux session on this machine.
#
# Usage: watch.local.sh [user@host] [port]
#        defaults to the pod recorded in ~/.msmarco9m/pod.json
set -uo pipefail
SESSION="${SESSION:-msmarco}"
STATE="$HOME/.msmarco9m/pod.json"

if [ $# -ge 2 ]; then
  TARGET="$1"; PORT="$2"
else
  # pod9m.sh writes the state file once the pod exposes ssh, which is minutes
  # after provisioning. Wait for it rather than failing.
  for _ in $(seq 1 90); do
    [ -f "$STATE" ] && break
    sleep 10
  done
  [ -f "$STATE" ] || { echo "no pod recorded after 15 min; is './pod9m.sh up' running?" >&2; exit 1; }
  TARGET="root@$(sed -n 's/.*"host": *"\([^"]*\)".*/\1/p' "$STATE")"
  PORT=$(sed -n 's/.*"port": *\([0-9]*\).*/\1/p' "$STATE")
fi

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o IdentitiesOnly=yes -o ServerAliveInterval=30 -i $HOME/.ssh/runpod_qa"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "attaching to existing session '$SESSION'"
  exec tmux attach -t "$SESSION"
fi

# -t forces a pty so the remote tmux can draw; without it the monitor exits with
# "open terminal failed: not a terminal".
#
# Wrapped in a reconnect loop: a dropped ssh would otherwise close the window and
# take the whole session with it, which is exactly what happens overnight on a
# laptop that sleeps. The remote tmux keeps running regardless, so reconnecting
# re-attaches to the same monitor.
tmux new-session -d -s "$SESSION" -x 220 -y 55 \
  "while :; do \
     ssh -t $SSH_OPTS -p $PORT $TARGET 'bash /root/monitor9m.sh'; \
     echo; echo '--- disconnected; reconnecting in 10s (ctrl-c to stop) ---'; \
     sleep 10; \
   done"
echo "session '$SESSION' created against $TARGET:$PORT"
echo "attach:  tmux attach -t $SESSION"
echo "detach:  ctrl-b d   (the run keeps going)"
exec tmux attach -t "$SESSION"
