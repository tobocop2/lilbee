#!/usr/bin/env bash
# Downloads the pinned golden artifacts ONCE (prep pod, Phase 2) into
# /workspace/golden. After this, every GPU pod installs fully offline via
# bootstrap.sh + SHA256SUMS. Runs on the SAME image as the job pods, so the
# apt dependency closure captured here is exactly what they need.
set -euo pipefail
G=/workspace/golden
mkdir -p "$G"/debs/partial "$G"/bin   # apt requires archives/partial
cd "$G"

fetch() { # url dest
  for i in 1 2 3; do curl -fsSL --retry 3 -o "$2" "$1" && return 0; sleep 5; done
  echo "FETCH_FAIL $1"; exit 1
}

fetch https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb debs/google-chrome.deb

# Only the pinned Chrome deb is staged; all other system packages install
# from apt at bootstrap time (offline closure curation was fragile — apt
# --download-only skips already-satisfied transitive deps).
[ -s debs/google-chrome.deb ] || { echo "FETCH_FAIL: chrome deb missing"; exit 1; }

fetch https://github.com/charmbracelet/vhs/releases/download/v0.10.0/vhs_0.10.0_Linux_x86_64.tar.gz /tmp/vhs.tgz
rm -rf /tmp/vhs-x && mkdir -p /tmp/vhs-x && tar -xzf /tmp/vhs.tgz -C /tmp/vhs-x
find /tmp/vhs-x -name vhs -type f -exec install -m0755 {} bin/vhs \;
[ -x bin/vhs ] || { echo "FETCH_FAIL: vhs binary not staged"; exit 1; }
fetch https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 bin/ttyd
chmod +x bin/ttyd

fetch https://github.com/JetBrains/JetBrainsMono/releases/download/v2.304/JetBrainsMono-2.304.zip /tmp/jbm.zip
rm -rf /tmp/jbm && unzip -qo /tmp/jbm.zip -d /tmp/jbm
fetch https://github.com/ryanoasis/nerd-fonts/releases/download/v3.2.1/NerdFontsSymbolsOnly.zip /tmp/sym.zip
rm -rf /tmp/sym && unzip -qo /tmp/sym.zip -d /tmp/sym
rm -rf /tmp/fonts-stage && mkdir -p /tmp/fonts-stage
cp /tmp/jbm/fonts/ttf/*.ttf /tmp/fonts-stage/
cp /tmp/sym/*.ttf /tmp/fonts-stage/
tar -cf fonts.tar -C /tmp/fonts-stage .
rm -rf /tmp/jbm* /tmp/sym* /tmp/fonts-stage /tmp/vhs.tgz /tmp/vhs-x

echo "FETCH_GOLDEN_OK ($NDEB debs)"
