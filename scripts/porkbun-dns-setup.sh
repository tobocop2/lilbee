#!/usr/bin/env bash
# Porkbun DNS setup for lilbee.sh.
#
# Configures DNS at Porkbun for the marketing site (apex, GitHub Pages) and
# the Obsidian plugin site (obsidian subdomain). Run once; idempotent.
#
# Credential resolution order:
#   1. pass: `pass show porkbun/tobocop/api-key` and `pass show porkbun/tobocop/api-secret`
#   2. env: PORKBUN_API_KEY and PORKBUN_SECRET_API_KEY
#   3. interactive prompt (read -s)
#
# Requires: curl, jq. (`brew install jq` on macOS.)

set -euo pipefail

DOMAIN="${DOMAIN:-lilbee.sh}"
API_BASE="https://api.porkbun.com/api/json/v3"
PASS_KEY_PATH="porkbun/tobocop/api-key"
PASS_SECRET_PATH="porkbun/tobocop/api-secret"
RECORD_TTL="600"

# GitHub Pages user-page host (CNAME target for project subdomains) and the
# anycast IPs (A records for the apex). Update if GH ever rotates them; see
# https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site
GH_PAGES_HOST="tobocop2.github.io"
GH_PAGES_IPS=(
  185.199.108.153
  185.199.109.153
  185.199.110.153
  185.199.111.153
)

# log goes to stderr so command-substitution stdout stays clean for get_secret.
log() { echo "$@" >&2; }

get_secret() {
  local label="$1" pass_path="$2" env_var="$3"

  if command -v pass >/dev/null && pass show "$pass_path" >/dev/null 2>&1; then
    log "  $label: read from pass ($pass_path)"
    pass show "$pass_path" | head -n 1
    return
  fi

  if [[ -n "${!env_var:-}" ]]; then
    log "  $label: read from \$$env_var"
    echo "${!env_var}"
    return
  fi

  local val
  read -r -s -p "$label: " val
  echo >&2
  echo "$val"
}

api() {
  local path="$1" body="${2:-$AUTH_BODY}"
  curl -sS -X POST -H "Content-Type: application/json" -d "$body" "$API_BASE$path"
}

check_status() {
  local resp="$1" label="$2"
  local status
  status=$(echo "$resp" | jq -r '.status // "UNKNOWN"')
  if [[ "$status" != "SUCCESS" ]]; then
    log "  ! $label: $(echo "$resp" | jq -r '.message // "no message"')"
    return 1
  fi
  log "  ok"
}

print_records() {
  api "/dns/retrieve/$DOMAIN" | jq -r '
    .records // []
    | .[]
    | "\(.type)\t\(.name)\t\(.content)\tttl=\(.ttl)"
  ' | column -t -s $'\t' >&2
}

create_record() {
  local name="$1" type="$2" content="$3"
  local body
  body=$(jq -n \
    --arg k "$API_KEY" --arg s "$SECRET_KEY" \
    --arg n "$name" --arg t "$type" --arg c "$content" \
    --arg ttl "$RECORD_TTL" \
    '{apikey:$k, secretapikey:$s, name:$n, type:$t, content:$c, ttl:$ttl}')
  check_status "$(api "/dns/create/$DOMAIN" "$body")" "create $type ${name:-@} -> $content"
}

command -v curl >/dev/null || { log "curl is required"; exit 1; }
command -v jq   >/dev/null || { log "jq is required (brew install jq)"; exit 1; }

log "==> Resolving credentials"
API_KEY="$(get_secret "API key"        "$PASS_KEY_PATH"    PORKBUN_API_KEY)"
SECRET_KEY="$(get_secret "Secret API key" "$PASS_SECRET_PATH" PORKBUN_SECRET_API_KEY)"

if [[ -z "$API_KEY" || -z "$SECRET_KEY" ]]; then
  log "Missing credentials"; exit 1
fi

AUTH_BODY=$(jq -n --arg k "$API_KEY" --arg s "$SECRET_KEY" \
  '{apikey:$k, secretapikey:$s}')

log
log "==> Verifying API credentials"
check_status "$(api /ping)" "ping" || exit 1

log
log "==> Current DNS records for $DOMAIN:"
print_records

log
read -r -p "Proceed: replace apex A/ALIAS and obsidian CNAME? [y/N] " confirm
[[ "$confirm" =~ ^[Yy]$ ]] || { log "aborted"; exit 0; }

log
log "==> Deleting existing records"
check_status "$(api "/dns/deleteByNameType/$DOMAIN/A")" "delete A @ apex" || true
check_status "$(api "/dns/deleteByNameType/$DOMAIN/ALIAS")" "delete ALIAS @ apex" || true
check_status "$(api "/dns/deleteByNameType/$DOMAIN/CNAME/obsidian")" "delete CNAME @ obsidian" || true

log
log "==> Creating 4 A records on apex (GitHub Pages)"
for ip in "${GH_PAGES_IPS[@]}"; do
  log "  $ip"
  create_record "" "A" "$ip"
done

log
log "==> Creating CNAME obsidian.$DOMAIN -> $GH_PAGES_HOST"
create_record "obsidian" "CNAME" "$GH_PAGES_HOST"

log
log "==> Resulting DNS records:"
print_records

cat >&2 <<EOF

Done. Wait 5-30 min for DNS to propagate, then verify externally:
  dig lilbee.sh +short             # should return the 4 GitHub IPs
  dig obsidian.lilbee.sh +short    # should return tobocop2.github.io, then IPs

Once both resolve correctly, push the CNAME files to both repos and update
the README / pyproject.toml URL references.
EOF
