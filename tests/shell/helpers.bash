#!/usr/bin/env bash
# Shared fixture builders for the release-script bats suites.

# Write the release-candidate job list the stubbed `gh api .../jobs` serves,
# plus the check-run annotations each of those jobs carries.
rc_jobs() {  # name:conclusion[:timeout|:user] ...
  local jobs="" entry name rest conclusion kind id=1000
  mkdir -p "${FIXTURE}/annotations"
  rm -f "${FIXTURE}/annotations"/*
  for entry in "$@"; do
    name="${entry%%:*}"
    rest="${entry#*:}"
    conclusion="${rest%%:*}"
    kind="none"
    [ "${rest}" != "${conclusion}" ] && kind="${rest#*:}"
    id=$(( id + 1 ))
    jobs="${jobs}${jobs:+,}{\"id\":${id},\"name\":\"${name}\",\"conclusion\":\"${conclusion}\"}"
    case "${kind}" in
      timeout)
        printf 'The job has exceeded the maximum execution time of 5h0m0s\nThe operation was canceled.\n' \
          > "${FIXTURE}/annotations/${id}"
        ;;
      user) printf 'The run was canceled by @someone.\n' > "${FIXTURE}/annotations/${id}" ;;
      stopped) printf 'The operation was canceled.\n' > "${FIXTURE}/annotations/${id}" ;;
      *) : > "${FIXTURE}/annotations/${id}" ;;
    esac
  done
  printf '{"jobs":[%s]}\n' "${jobs}" > "${FIXTURE}/rc_jobs.json"
}
