# Golden render environment. Mirrors what bootstrap.sh installs on pods from
# /workspace/golden (same pinned versions; sources in golden_sources.txt).
# Base matches the runpod pytorch image's OS (ubuntu 22.04).
FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl unzip fontconfig ffmpeg tesseract-ocr tesseract-ocr-eng tmux \
    fonts-noto-color-emoji python3 python3-pip zstd bash zsh \
    && rm -rf /var/lib/apt/lists/*

# Chrome (pinned deb kept in golden set; latest at draft time). Root +
# sandbox is refused by Chrome, so --no-sandbox is baked into the wrapper —
# environment-independent, matches what pods get via bootstrap.sh.
RUN curl -fsSL -o /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && apt-get update && apt-get install -y /tmp/chrome.deb && rm -rf /var/lib/apt/lists/* /tmp/chrome.deb \
    && sed -i 's|exec -a "$0" "$HERE/chrome" "$@"|exec -a "$0" "$HERE/chrome" "$@" --no-sandbox|' /opt/google/chrome/google-chrome

# vhs 0.10.0 + ttyd 1.7.7 (pinned)
RUN curl -fsSL https://github.com/charmbracelet/vhs/releases/download/v0.10.0/vhs_0.10.0_Linux_x86_64.tar.gz \
      | tar -xz -C /tmp && install -m0755 /tmp/vhs*/vhs /usr/local/bin/vhs 2>/dev/null \
      || install -m0755 /tmp/vhs /usr/local/bin/vhs
RUN curl -fsSL -o /usr/local/bin/ttyd https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 \
    && chmod +x /usr/local/bin/ttyd

# Fonts: JetBrains Mono (all weights) + Symbols Nerd Font Mono
RUN mkdir -p /usr/share/fonts/reels \
    && curl -fsSL -o /tmp/jbm.zip https://github.com/JetBrains/JetBrainsMono/releases/download/v2.304/JetBrainsMono-2.304.zip \
    && unzip -q /tmp/jbm.zip -d /tmp/jbm && cp /tmp/jbm/fonts/ttf/*.ttf /usr/share/fonts/reels/ \
    && curl -fsSL -o /tmp/sym.zip https://github.com/ryanoasis/nerd-fonts/releases/download/v3.2.1/NerdFontsSymbolsOnly.zip \
    && unzip -q /tmp/sym.zip -d /tmp/sym && cp /tmp/sym/*.ttf /usr/share/fonts/reels/ \
    && fc-cache -f && rm -rf /tmp/jbm* /tmp/sym*

RUN pip3 install --no-cache-dir numpy pillow pyyaml

WORKDIR /work
