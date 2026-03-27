#!/usr/bin/env bash
# Audit script for litellm supply-chain attack (versions 1.82.7 / 1.82.8)
# Based on: https://www.xda-developers.com/popular-python-library-backdoor-machine/
#
# The malware was a 3-stage credential stealer:
#   1. .pth file (runs on ANY python startup) -> drops payload
#   2. Payload at ~/.config/sysmon/sysmon.py -> steals creds, exfils to models.litellm.cloud
#   3. Persistence via systemd service + k8s pod backdoor
#
# Exit codes: 0 = clean, 1 = indicators found

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

found=0

warn() { echo -e "${RED}[!] FOUND:${NC} $1"; found=1; }
ok()   { echo -e "${GREEN}[✓]${NC} $1"; }
info() { echo -e "${YELLOW}[~]${NC} $1"; }
section() { echo -e "\n${BOLD}── $1 ──${NC}"; }

# ─── Stage 1: Malicious .pth file ───────────────────────────────────
section "Stage 1: Malicious .pth loader"

# Check all Python site-packages dirs for the .pth trigger
pth_found=false
while IFS= read -r sp; do
    if [ -f "$sp/litellm_init.pth" ]; then
        warn "Malicious .pth file: $sp/litellm_init.pth"
        pth_found=true
    fi
done < <(python3 -c "import site; print('\n'.join(site.getsitepackages() + [site.getusersitepackages()]))" 2>/dev/null || true)

# Also check common venv locations in the project
for venv in .venv venv env; do
    if [ -d "$venv" ]; then
        while IFS= read -r f; do
            warn "Malicious .pth file: $f"
            pth_found=true
        done < <(find "$venv" -name "litellm_init.pth" 2>/dev/null)
    fi
done

$pth_found || ok "No litellm_init.pth found in any site-packages"

# ─── Stage 2: Backdoor payload ──────────────────────────────────────
section "Stage 2: Backdoor payload"

if [ -f "$HOME/.config/sysmon/sysmon.py" ]; then
    warn "Backdoor payload: ~/.config/sysmon/sysmon.py"
    echo "  Contents (first 20 lines):"
    head -20 "$HOME/.config/sysmon/sysmon.py" | sed 's/^/    /'
elif [ -d "$HOME/.config/sysmon" ]; then
    warn "Suspicious directory exists: ~/.config/sysmon/"
    ls -la "$HOME/.config/sysmon/" | sed 's/^/    /'
else
    ok "No backdoor payload at ~/.config/sysmon/"
fi

# ─── Stage 3: Persistence mechanisms ────────────────────────────────
section "Stage 3: Persistence"

# Systemd service for the backdoor
sysmon_svc=false
for dir in "$HOME/.config/systemd/user" /etc/systemd/system /usr/lib/systemd/system; do
    for name in sysmon sysmon-monitor; do
        if [ -f "$dir/$name.service" ]; then
            warn "Systemd service: $dir/$name.service"
            sysmon_svc=true
        fi
    done
done
$sysmon_svc || ok "No sysmon systemd services found"

# Check if sysmon service is active
if systemctl --user is-active sysmon.service &>/dev/null; then
    warn "sysmon.service is RUNNING (user scope)"
fi
if systemctl is-active sysmon.service &>/dev/null 2>&1; then
    warn "sysmon.service is RUNNING (system scope)"
fi

# Kubernetes backdoor (privileged pod in kube-system)
if command -v kubectl &>/dev/null; then
    if kubectl get pods -n kube-system -o name 2>/dev/null | grep -qi "litellm\|sysmon\|backdoor"; then
        warn "Suspicious pod found in kube-system namespace"
    else
        ok "No suspicious k8s pods in kube-system"
    fi
else
    info "kubectl not found — skipping k8s check"
fi

# ─── Network: C2 domain ─────────────────────────────────────────────
section "Network indicators"

c2_domain="models.litellm.cloud"

# Check DNS resolution
if host "$c2_domain" &>/dev/null 2>&1 || dig +short "$c2_domain" 2>/dev/null | grep -q .; then
    info "C2 domain $c2_domain resolves (may still be up)"
else
    ok "C2 domain $c2_domain does not resolve"
fi

# Check for active connections to C2
if command -v lsof &>/dev/null; then
    if lsof -i -n 2>/dev/null | grep -i "litellm\|sysmon" | grep -v grep; then
        warn "Active connections to suspicious processes"
    else
        ok "No active connections to litellm/sysmon processes"
    fi
fi

# Check DNS cache / recent connections (macOS)
if [ "$(uname)" = "Darwin" ]; then
    if log show --predicate 'process == "mDNSResponder"' --last 24h 2>/dev/null | grep -q "$c2_domain"; then
        warn "DNS query to $c2_domain found in last 24h"
    else
        ok "No DNS queries to $c2_domain in last 24h"
    fi
fi

# ─── Credential exposure check ──────────────────────────────────────
section "Credential exposure assessment"
info "The malware targeted these — review/rotate if you were compromised:"

check_exists() {
    local label="$1" path="$2"
    if [ -e "$path" ]; then
        echo -e "  ${YELLOW}EXISTS${NC}: $label ($path)"
    fi
}

echo ""
echo "  SSH keys:"
for key in id_rsa id_ed25519 id_ecdsa id_dsa; do
    check_exists "  $key" "$HOME/.ssh/$key"
done

echo "  Git config:"
check_exists "  .gitconfig" "$HOME/.gitconfig"
check_exists "  git-credentials" "$HOME/.git-credentials"

echo "  Cloud credentials:"
check_exists "  AWS credentials" "$HOME/.aws/credentials"
check_exists "  AWS config" "$HOME/.aws/config"
check_exists "  GCP service account" "$HOME/.config/gcloud/application_default_credentials.json"
check_exists "  Azure config" "$HOME/.azure/"
check_exists "  Kubernetes config" "$HOME/.kube/config"

echo "  Docker:"
check_exists "  Docker config" "$HOME/.docker/config.json"

echo "  Databases:"
check_exists "  PostgreSQL pass" "$HOME/.pgpass"
check_exists "  MySQL config" "$HOME/.my.cnf"

echo "  Tokens in environment:"
# Check if common secret env vars are set (don't print values!)
for var in AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN OPENAI_API_KEY ANTHROPIC_API_KEY \
           GITHUB_TOKEN GH_TOKEN SLACK_TOKEN DISCORD_WEBHOOK_URL DATABASE_URL \
           REDIS_URL LDAP_PASSWORD; do
    if [ -n "${!var+x}" ] 2>/dev/null; then
        echo -e "  ${YELLOW}SET${NC}: \$$var — rotate this key if compromised"
    fi
done

# ─── Installed litellm version check ─────────────────────────────────
section "Installed litellm versions"

# Check all discoverable Python environments
while IFS= read -r python_bin; do
    ver=$("$python_bin" -c "import importlib.metadata; print(importlib.metadata.version('litellm'))" 2>/dev/null) || continue
    case "$ver" in
        1.82.7|1.82.8)
            warn "COMPROMISED litellm $ver in: $python_bin"
            ;;
        *)
            ok "litellm $ver in: $python_bin"
            ;;
    esac
done < <(which -a python3 python 2>/dev/null; find . -path "*/bin/python*" -maxdepth 4 2>/dev/null)

# ─── Summary ─────────────────────────────────────────────────────────
section "Summary"
if [ "$found" -gt 0 ]; then
    echo -e "${RED}${BOLD}INDICATORS FOUND — review warnings above and take action:${NC}"
    echo "  1. Remove any malicious files identified"
    echo "  2. Kill sysmon processes: pkill -f sysmon.py"
    echo "  3. Remove systemd services: systemctl --user disable sysmon"
    echo "  4. Rotate ALL credentials listed above"
    echo "  5. Check git log for unauthorized commits"
    echo "  6. Audit outbound network logs for $c2_domain"
    exit 1
else
    echo -e "${GREEN}${BOLD}No indicators of compromise found.${NC}"
    echo "  Your litellm versions pre-date the attack."
    echo "  Credentials listed above were NOT exposed."
    exit 0
fi
