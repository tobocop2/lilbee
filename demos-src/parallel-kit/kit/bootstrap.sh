#!/usr/bin/env bash
# Offline pod bootstrap from the golden artifact set on the volume.
# Deterministic: every artifact verified against SHA256SUMS before install;
# no network fetches. Idempotent. ~60-90s.
set -euo pipefail
GOLDEN="${GOLDEN:-/workspace/golden}"

[ -d "$GOLDEN" ] || { echo "BOOTSTRAP_FAIL: $GOLDEN missing"; exit 1; }
# Integrity-check only the artifacts we actually consume (pinned binaries,
# fonts, chrome deb, runtime tarballs) — NOT the apt closure.
( cd "$GOLDEN" && grep -E 'bin/|fonts.tar|tar.zst|tar.gz|debs/google-chrome' SHA256SUMS | sha256sum -c --quiet ) \
  || { echo "BOOTSTRAP_FAIL: artifact checksum mismatch"; exit 1; }

export DEBIAN_FRONTEND=noninteractive
# System packages come from apt (pods have network — same path that pulls the
# base image and hf models). Offline deb-closure curation was fragile: apt
# --download-only skips already-satisfied transitive deps, so fresh pods
# missed libevent-core/libutempter and tmux would not configure.
apt-get update -qq || { echo "BOOTSTRAP_FAIL: apt update"; exit 1; }
apt-get install -y -qq fontconfig ffmpeg tesseract-ocr tesseract-ocr-eng \
  tmux zstd zsh fonts-noto-color-emoji || { echo "BOOTSTRAP_FAIL: apt install"; exit 1; }
apt-get install -y -qq "$GOLDEN/debs/google-chrome.deb" \
  || { echo "BOOTSTRAP_FAIL: chrome install"; exit 1; }
# Chrome refuses root+sandbox; bake --no-sandbox into the wrapper (same as
# the golden Docker image) and verify it took.
sed -i 's|exec -a "$0" "$HERE/chrome" "$@"$|exec -a "$0" "$HERE/chrome" "$@" --no-sandbox|' /opt/google/chrome/google-chrome
grep -q -- '--no-sandbox' /opt/google/chrome/google-chrome || { echo "BOOTSTRAP_FAIL: chrome wrapper sed missed"; exit 1; }

# Fonts: JetBrains Mono (4 weights), Symbols Nerd Font Mono (+ Noto emoji deb)
mkdir -p /usr/share/fonts/reels
tar -xf "$GOLDEN/fonts.tar" -C /usr/share/fonts/reels
fc-cache -f >/dev/null

install -m 0755 "$GOLDEN/bin/vhs" /usr/local/bin/vhs
install -m 0755 "$GOLDEN/bin/ttyd" /usr/local/bin/ttyd

# Self-contained runtime bundles -> local NVMe (never run python off the volume)
tar --zstd -xf "$GOLDEN/venv.tar.zst" -C /root      # /root/venv (main lilbee + qa deps)
tar --zstd -xf "$GOLDEN/agents.tar.zst" -C /         # /root/.opencode + hermes FHS layout
tar --zstd -xf "$GOLDEN/engine-multiarch.tar.zst" -C /root   # archive root is engine/ -> /root/engine

for bin in /root/engine/llama-server /root/engine/llama-swap \
           /root/.opencode/bin/opencode /usr/local/bin/hermes; do
  [ -x "$bin" ] || { echo "BOOTSTRAP_FAIL: $bin missing"; exit 1; }
done

echo "BOOTSTRAP_OK $(date -u +%H:%M:%S)"
