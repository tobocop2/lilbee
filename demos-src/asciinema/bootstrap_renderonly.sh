#!/usr/bin/env bash
# Render-stack-only bootstrap (prep pod / render iteration): debs, fonts,
# vhs/ttyd from the golden set. GPU pods use the full bootstrap.sh instead.
set -euo pipefail
GOLDEN="${GOLDEN:-/workspace/golden}"
export DEBIAN_FRONTEND=noninteractive
dpkg -i --skip-same-version "$GOLDEN"/debs/*.deb >/dev/null 2>&1 || apt-get -y -f install >/dev/null
sed -i 's|exec -a "$0" "$HERE/chrome" "$@"$|exec -a "$0" "$HERE/chrome" "$@" --no-sandbox|' /opt/google/chrome/google-chrome
mkdir -p /usr/share/fonts/reels
tar -xf "$GOLDEN/fonts.tar" -C /usr/share/fonts/reels
fc-cache -f >/dev/null
install -m0755 "$GOLDEN/bin/vhs" /usr/local/bin/vhs
install -m0755 "$GOLDEN/bin/ttyd" /usr/local/bin/ttyd
echo "RENDER_STACK_OK"
