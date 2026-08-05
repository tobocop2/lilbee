#!/usr/bin/env bash
# Billing backstop: terminate this pod if the reel job disappears or the pod
# idles far beyond any plausible take. Never fires during normal operation —
# job.sh terminates first.
set -u
API_KEY_FILE=/root/.runpod_key
RUNPOD_POD_ID="${RUNPOD_POD_ID:?}"
HARD_DEADLINE_S=${HARD_DEADLINE_S:-14400}   # 4h absolute cap per pod
NO_JOB_GRACE_S=${NO_JOB_GRACE_S:-900}       # job.sh gone for 15 min -> kill

t0=$(date +%s)
no_job_since=""
while true; do
  sleep 60
  now=$(date +%s)
  if [ $((now - t0)) -gt "$HARD_DEADLINE_S" ]; then
    reason="hard deadline"; break
  fi
  if pgrep -f '[j]ob.sh' >/dev/null; then
    no_job_since=""
  else
    [ -z "$no_job_since" ] && no_job_since=$now
    if [ $((now - no_job_since)) -gt "$NO_JOB_GRACE_S" ]; then
      reason="job.sh gone"; break
    fi
  fi
done

echo "watchdog terminating pod: $reason"
curl -s -X POST "https://api.runpod.io/graphql?api_key=$(cat $API_KEY_FILE)" \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)' \
  -H 'Content-Type: application/json' \
  -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}"
