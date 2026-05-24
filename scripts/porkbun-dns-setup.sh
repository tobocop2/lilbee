#!/usr/bin/env bash
# Porkbun DNS setup for lilbee.sh.
#
# Configures DNS at Porkbun for the marketing site (apex, GitHub Pages),
# the Obsidian plugin site (obsidian subdomain), and a wildcard URL forward
# that redirects any other subdomain to the canonical https://lilbee.sh.
# Run once; idempotent.
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

# Subdomains that resolve to GitHub Pages projects via CNAME. Each repo's
# own site/CNAME file tells GH which repo to serve for each host. 'www' is
# an apex alias whose CNAME lets GH issue a SAN HTTPS cert covering both
# lilbee.sh and www.lilbee.sh.
GH_PAGES_SUBDOMAINS=(obsidian www)

# Wildcard URL forward: any *.lilbee.sh subdomain that isn't otherwise defined
# 301-redirects to the canonical site, preserving the request path.
WILDCARD_REDIRECT_TARGET="https://$DOMAIN"

# Apex TXT records to ensure (site verification, etc.). Additive and idempotent:
# ensure_apex_txt creates one only when an identical record is missing, and the
# delete phase below never touches TXT, so existing SPF/DKIM records are safe.
APEX_TXT_RECORDS=(
  "google-site-verification=443jGjxV6o1LGFRxlgixDTOOQ3mnUO0FwwlEwnH96Uc"  # Google Search Console
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

print_forwards() {
  api "/domain/getUrlForwarding/$DOMAIN" | jq -r --arg dom "$DOMAIN" '
    .forwards // []
    | .[]
    | "\(if .wildcard == "yes" then "*" else (.subdomain // "") end).\($dom)\t\(.type)\t-> \(.location)\tincludePath=\(.includePath)"
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

# Create an apex TXT record only if an identical one isn't already present, so
# re-runs don't stack duplicates and other TXT records (SPF, DKIM) are untouched.
ensure_apex_txt() {
  local content="$1" existing
  existing=$(api "/dns/retrieve/$DOMAIN" \
    | jq -r --arg c "$content" '.records[]? | select(.type == "TXT" and .content == $c) | .id' \
    | head -n 1)
  if [[ -n "$existing" ]]; then
    log "  exists (id=$existing): $content"
    return 0
  fi
  create_record "" "TXT" "$content"
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
log "==> Current URL forwards for $DOMAIN:"
print_forwards

log
read -r -p "Proceed: replace records + wildcard forward to $WILDCARD_REDIRECT_TARGET? [y/N] " confirm
[[ "$confirm" =~ ^[Yy]$ ]] || { log "aborted"; exit 0; }

log
log "==> Deleting existing records"
check_status "$(api "/dns/deleteByNameType/$DOMAIN/A")" "delete A @ apex" || true
check_status "$(api "/dns/deleteByNameType/$DOMAIN/ALIAS")" "delete ALIAS @ apex" || true
for subdomain in "${GH_PAGES_SUBDOMAINS[@]}"; do
  check_status "$(api "/dns/deleteByNameType/$DOMAIN/CNAME/$subdomain")" "delete CNAME @ $subdomain" || true
done

log "==> Deleting existing wildcard CNAME (Porkbun parking default)"
api "/dns/retrieve/$DOMAIN" | jq -r '.records[]? | select(.type == "CNAME" and (.name | startswith("*."))) | .id' | while IFS= read -r record_id; do
  [[ -z "$record_id" ]] && continue
  check_status "$(api "/dns/delete/$DOMAIN/$record_id")" "delete wildcard CNAME id=$record_id" || true
done

log "==> Deleting existing wildcard URL forwards"
api "/domain/getUrlForwarding/$DOMAIN" | jq -r '.forwards[]? | select(.wildcard == "yes") | .id' | while IFS= read -r forward_id; do
  [[ -z "$forward_id" ]] && continue
  check_status "$(api "/domain/deleteUrlForward/$DOMAIN/$forward_id")" "delete URL forward id=$forward_id" || true
done

log
log "==> Creating ${#GH_PAGES_IPS[@]} A records on apex (GitHub Pages)"
for ip in "${GH_PAGES_IPS[@]}"; do
  log "  $ip"
  create_record "" "A" "$ip"
done

for subdomain in "${GH_PAGES_SUBDOMAINS[@]}"; do
  log
  log "==> Creating CNAME $subdomain.$DOMAIN -> $GH_PAGES_HOST"
  create_record "$subdomain" "CNAME" "$GH_PAGES_HOST"
done

log
log "==> Creating wildcard URL forward *.$DOMAIN -> $WILDCARD_REDIRECT_TARGET (permanent, include path)"
forward_body=$(jq -n \
  --arg k "$API_KEY" --arg s "$SECRET_KEY" \
  --arg loc "$WILDCARD_REDIRECT_TARGET" \
  '{apikey:$k, secretapikey:$s, subdomain:"", location:$loc, type:"permanent", includePath:"yes", wildcard:"yes"}')
check_status "$(api "/domain/addUrlForward/$DOMAIN" "$forward_body")" "create wildcard URL forward"

log
log "==> Ensuring apex TXT records (verification; additive, idempotent)"
for txt in "${APEX_TXT_RECORDS[@]}"; do
  ensure_apex_txt "$txt"
done

log
log "==> Resulting DNS records:"
print_records

log
log "==> Resulting URL forwards:"
print_forwards

cat >&2 <<EOF

Done. Wait 5-30 min for DNS to propagate, then verify externally:
  dig lilbee.sh +short              # 4 GitHub IPs
  dig www.lilbee.sh +short          # CNAME chain through tobocop2.github.io
  dig obsidian.lilbee.sh +short     # CNAME chain through tobocop2.github.io
  curl -I http://anything.lilbee.sh # 301 to https://lilbee.sh/ via Porkbun forward

The repo's site/CNAME files activate the custom domains on the next Pages
deploy. After GitHub provisions HTTPS certs (5-30 min post-deploy), enable
"Enforce HTTPS" in each repo's Pages settings. The www alias gets covered
by the same SAN cert as the apex automatically.

Note: HTTPS to wildcard subdomains (https://foo.lilbee.sh) returns a cert
error because Porkbun's URL forwarder doesn't ship a wildcard cert. HTTP
works and redirects cleanly. Specific subdomains (www, obsidian) get real
GH Pages HTTPS.
EOF
