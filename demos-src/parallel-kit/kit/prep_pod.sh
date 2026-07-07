#!/usr/bin/env bash
# Phase 2: prep pod (cheap GPU, US-CA-2, volume mounted, same image as job
# pods). Refreshes the volume for main-branch recording, stages the golden
# artifact set, and runs the AUTHORITATIVE render qualification on real
# amd64. Steps with DONE markers skip on re-run; the tail steps (reconcile,
# qualgate, pack, checksum) ALWAYS run so re-synced tapes are never stale.
# Run inside tmux; ~40-60 min total.
set -Eeuo pipefail
cd /workspace
mkdir -p golden reels-out qa
DONE=/workspace/golden/.done
mkdir -p "$DONE"
LLAMA_CPP_PIN="${LLAMA_CPP_PIN:-b6660}"   # nearest release tag to the proven 6658925 build

CURRENT_STEP=""
trap 'echo "STEP_FAIL ${CURRENT_STEP}"' ERR

step() { # step <name> <fn> — errexit stays ACTIVE inside fn (no && wrapper)
  CURRENT_STEP="$1"
  if [ -f "$DONE/$1" ]; then echo "SKIP $1"; return 0; fi
  echo "== $1 =="
  "$2"
  touch "$DONE/$1"
}

always() { # always <name> <fn> — never skipped
  CURRENT_STEP="$1"
  echo "== $1 (always) =="
  "$2"
}

seed_fresh_volume() {
  # Fresh-DC volume: every source is reconstructible. No-ops where content
  # already exists (US-CA-2 volumes skip almost all of this).
  cd /workspace
  [ -d lilbee/.git ] || git clone https://github.com/tobocop2/lilbee /workspace/lilbee
  if [ ! -d corpus ] || [ -z "$(ls -A corpus 2>/dev/null)" ]; then
    mkdir -p corpus && cd lilbee && find src/lilbee/providers src/lilbee/data \
      src/lilbee/server src/lilbee/app -name '*.py' -exec cp {} /workspace/corpus/ \; && cd /workspace
  fi
  if [ ! -d godot/doc/classes ]; then
    rm -rf godot && git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/godotengine/godot /workspace/godot
    ( cd godot && git sparse-checkout set doc/classes )
  fi
  [ -f cv-manual.pdf ] || { echo "SEED_FAIL: cv-manual.pdf missing (scatter uploads it)"; exit 1; }
  mkdir -p /workspace/engine-cache/cu124
  if [ ! -x /workspace/engine-cache/cu124/llama-swap ]; then
    curl -fsSL -o /tmp/lswap.tar.gz \
      https://github.com/mostlygeek/llama-swap/releases/download/v235/llama-swap_235_linux_amd64.tar.gz \
      && tar -xzf /tmp/lswap.tar.gz -C /tmp \
      && install -m0755 /tmp/llama-swap /workspace/engine-cache/cu124/llama-swap
  fi
  if [ ! -x /workspace/engine-cache/cu124/gguf-parser ]; then
    curl -fsSL -o /tmp/gguf-parser-linux-amd64 \
      https://github.com/gpustack/gguf-parser-go/releases/download/v0.24.1/gguf-parser-linux-amd64 \
      && install -m0755 /tmp/gguf-parser-linux-amd64 /workspace/engine-cache/cu124/gguf-parser
  fi
  if ! ls /workspace/models/models--*Embedding* >/dev/null 2>&1; then
    pip install -q -U "huggingface_hub[cli]" 2>/dev/null || pip install -q huggingface_hub
    HF_HOME=/workspace/hfseed hf download Qwen/Qwen3-Embedding-8B-GGUF --include "*Q8_0*" \
      || HF_HOME=/workspace/hfseed huggingface-cli download Qwen/Qwen3-Embedding-8B-GGUF --include "*Q8_0*"
    mkdir -p /workspace/models
    mv /workspace/hfseed/hub/models--Qwen--Qwen3-Embedding-8B-GGUF /workspace/models/
  fi
  # models live per-pod (lilbee model pull); the volume never carries them
  echo '{"mode": "hf", "measured_MBps": 0}' > /workspace/golden/transfer.json
  touch "$DONE/measure_throughput"   # measurement is meaningless with no volume models
}

refresh_checkout() {
  cd /workspace/lilbee
  git fetch origin main && git checkout -f origin/main
  git rev-parse HEAD > /workspace/golden/lilbee_sha
  cd /workspace
  tar --zstd --exclude=.git -cf /workspace/golden/lilbee-src.tar.zst -C /workspace lilbee
}

stage_golden() {
  bash /workspace/v2/kit/fetch_golden.sh
  cp ~/.ssh/authorized_keys /workspace/golden/authorized_keys
  rm -rf /workspace/golden/godot-project /workspace/golden/godot-project-bare
  cp -r /workspace/v2/assets/godot-with-lilbee /workspace/golden/godot-project
  cp -r /workspace/v2/assets/godot-without-lilbee /workspace/golden/godot-project-bare
}

install_render_stack() {
  GOLDEN=/workspace/golden bash /workspace/v2/kit/bootstrap_renderonly.sh
}

rebuild_venv() {
  # self-contained venv at /root/venv (non-editable install so the tarball
  # works at the same path on every pod) + the qa-script deps
  python3 -m venv /root/venv
  /root/venv/bin/pip install -q --upgrade pip
  # lilbee-engine resolves via [tool.uv.sources] (in-repo path) which plain
  # pip ignores; install it from the repo first. Its empty bin/ is fine —
  # LILBEE_LLAMA_SERVER_PATH points at our multi-arch build.
  /root/venv/bin/pip install -q /workspace/lilbee/packaging/engine-wheel
  /root/venv/bin/pip install -q /workspace/lilbee numpy pillow pyyaml
  /root/venv/bin/lilbee --version
  /root/venv/bin/python3 -c "import numpy, PIL, yaml"
  tar --zstd -cf /workspace/golden/venv.tar.zst -C /root venv
}

build_engine() {
  # multi-arch llama-server: the cached sm_90-only build fails on every
  # A40/4090/A100 pod. Pinned tag for re-run determinism.
  command -v cmake >/dev/null || { apt-get update -qq && apt-get install -y -qq cmake build-essential; }
  export PATH=/usr/local/cuda/bin:$PATH CUDACXX=/usr/local/cuda/bin/nvcc   # nvcc absent from non-interactive PATH
  [ -d /workspace/llama.cpp ] || git clone https://github.com/ggerganov/llama.cpp /workspace/llama.cpp
  cd /workspace/llama.cpp
  git fetch --tags -q && git checkout -q "$LLAMA_CPP_PIN"
  cmake -B build-multi -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
        -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF 2>&1 | tail -1
  cmake --build build-multi -j"$(nproc)" --target llama-server 2>&1 | tail -1
  mkdir -p /root/engine
  cp build-multi/bin/llama-server /root/engine/
  cp /workspace/engine-cache/cu124/llama-swap /root/engine/
  cp /workspace/engine-cache/cu124/gguf-parser /root/engine/
  for bin in llama-server llama-swap gguf-parser; do [ -x /root/engine/$bin ]; done
  tar --zstd -cf /workspace/golden/engine-multiarch.tar.zst -C /root engine
  cd /workspace
}

reextract_kbs() {
  # main's kreuzberg 4.9 extractor; embedding runs on this pod's GPU via the
  # multi-arch engine (also proves the build on this pod's arch pre-fan-out)
  export PATH=/root/venv/bin:/root/engine:$PATH
  export LILBEE_LLAMA_SERVER_PATH=/root/engine/llama-server
  export LILBEE_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf"
  # embedder served from local NVMe (MFS reads are the slow path)
  mkdir -p /root/models-prep
  for d in /workspace/models/models--*; do
    case "$(basename "$d" | tr 'A-Z' 'a-z')" in *embed*) cp -r "$d" /root/models-prep/ ;; esac
  done
  export LILBEE_MODELS_DIR=/root/models-prep
  for kb in manual corpus godot; do
    W=/root/kbwork-$kb
    rm -rf "$W" && mkdir -p "$W/data/lancedb" "$W/documents"
    # add straight from the source: lilbee add copies into $W/documents
    # itself; pre-copying double-indexed every document
    case $kb in
      manual) SRC=/workspace/cv-manual.pdf ;;
      corpus) SRC=/workspace/corpus ;;
      godot)  SRC=/workspace/godot/doc/classes ;;
    esac
    LILBEE_DATA="$W" lilbee add "$SRC" 2>&1 | tail -2
    mkdir -p /root/kbsnap && rm -rf /root/kbsnap/kb-$kb-*
    cp -r "$W/documents" /root/kbsnap/kb-$kb-docs
    cp -r "$W/data/lancedb" /root/kbsnap/kb-$kb-lancedb
    tar --zstd -cf /workspace/golden/kb-$kb.tar.zst -C /root/kbsnap kb-$kb-docs kb-$kb-lancedb
  done
  # search sanity vs ground truth (full graded ask happens in canary).
  # `lilbee search` needs the embed server up, so run it inside a serve.
  export LILBEE_DATA=/root/kbwork-manual
  nohup lilbee serve > /root/sanity-serve.log 2>&1 &
  SERVE_PID=$!
  ok=""
  for i in $(seq 1 60); do
    sleep 5
    probe=$(lilbee search "trailer" --top-k 1 2>&1) && \
      ! echo "$probe" | grep -qi "no embed model server" && { ok=1; break; }
  done
  [ -n "$ok" ] || { kill $SERVE_PID 2>/dev/null; echo "SANITY_SERVE_NEVER_READY"; exit 1; }
  rc=0
  python3 /workspace/v2/kit/search_sanity.py /workspace/v2/kit/ground_truth.json || rc=$?
  kill $SERVE_PID 2>/dev/null; pkill -9 -x llama-server 2>/dev/null; pkill -9 -x llama-swap 2>/dev/null
  [ "$rc" = 0 ]
}

bundle_agents() {
  # opencode -> /root/.opencode; hermes uses an FHS root layout:
  # /usr/local/bin/hermes (wrapper) + /usr/local/lib/hermes-agent (self-
  # contained venv on the image's /usr/bin/python3.11) + /root/.hermes (data).
  # Installers/`--version` open a first-run wizard on an interactive stdin, so
  # every invocation gets </dev/null. Skip re-download when already present.
  [ -x /root/.opencode/bin/opencode ] || curl -fsSL https://opencode.ai/install | bash </dev/null
  /root/.opencode/bin/opencode --version </dev/null
  [ -x /usr/local/bin/hermes ] || curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash </dev/null
  hermes --version </dev/null
  # captured with -C / (absolute FHS paths); bootstrap extracts with -C /
  tar --zstd -cf /workspace/golden/agents.tar.zst -C / \
    root/.opencode root/.hermes usr/local/bin/hermes usr/local/lib/hermes-agent
}

resolve_pull_refs() {
  # registry refs are not derivable from (repo, quant) — quant may be a
  # subdir, a filename token, or absent. Resolve each model's exact
  # installed ref from the registry itself and smoke-test one pull.
  PATH=/root/venv/bin:$PATH LILBEE_MODELS_DIR=/workspace/models \
    python3 /workspace/v2/kit/resolve_pull_refs.py \
    /workspace/v2/reels.yaml /workspace/v2/kit/pull_refs.json
}

vram_table() {
  PATH=/root/venv/bin:/root/engine:$PATH python3 /workspace/v2/kit/vram_table.py \
    /workspace/v2/reels.yaml /workspace/v2/kit/vram_table.json
}

measure_throughput() {
  # O_DIRECT bypasses the page cache (reextract already read some GGUFs);
  # measure the largest file for a realistic heavy-pod copy rate
  f=$(find /workspace/models -name '*.gguf' -printf '%s %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
  t0=$(date +%s)
  dd if="$f" of=/dev/null bs=64M count=80 iflag=direct 2>/dev/null
  dt=$(( $(date +%s) - t0 )); [ "$dt" -gt 0 ] || dt=1
  mbps=$(( 80 * 64 / dt ))
  mode=volume; [ "$mbps" -lt 150 ] && mode=hf
  echo "{\"mode\": \"$mode\", \"measured_MBps\": $mbps}" > /workspace/golden/transfer.json
  cat /workspace/golden/transfer.json
}

geo_recalibrate() {
  # the packed geometry_cal.json must come from REAL amd64 Chrome/freetype,
  # not the Rosetta dev-loop solve; tapes are regenerated right after
  PATH=/root/venv/bin:$PATH python3 /workspace/v2/kit/geo_calibrate.py \
    /workspace/v2/reels.yaml /workspace/v2/kit/geometry_cal.json
}

reconcile_tapes() {
  # volume v8 bodies win for reels marked volume:; regenerate all tapes in
  # the v2 working tree (/workspace/v2, synced from the Mac)
  cd /workspace/v2
  for r in reel1-selfindex reel2-placement code-search; do
    src=$(python3 -c "import yaml; print(yaml.safe_load(open('reels.yaml'))['reels']['$r']['choreography'].split(':',1)[1])")
    [ -f "$src" ] && cp "$src" "tapes/src/$r.body.tape" || echo "WARN: $src missing on volume; gen_tapes will fail unless a synced body exists"
  done
  python3 kit/reconcile_windows.py reels.yaml /workspace/reels || true
  PATH=/root/venv/bin:$PATH bash gen_tapes.sh
  cd /workspace
}

authoritative_qualgate() {
  # temporary pack so qualgate runs against the exact runtime layout
  bash /workspace/v2/kit/pack_kit.sh /workspace/v2 /workspace/kit
  KIT=/workspace/kit QUALGATE_WORK=/root/qualgate PATH=/root/venv/bin:$PATH \
    bash /workspace/kit/qualgate.sh
  python3 - <<'PY'
import json
r = json.load(open('/root/qualgate/probe-report.json'))
ink = r['regular_ink']
json.dump({"regular_ink_band": [round(ink*0.65,4), round(ink*1.45,4)]},
          open('/workspace/v2/kit/calibration.json','w'))
print("calibration:", ink)
PY
  cp /root/qualgate/probe-glyphs.png /workspace/qa/authoritative-probe.png
  cp /root/qualgate/probe-glyphs.png /workspace/reels-out/authoritative-probe.png
}

final_pack_and_sums() {
  # repack WITH calibration.json, then checksum EVERYTHING pods consume
  bash /workspace/v2/kit/pack_kit.sh /workspace/v2 /workspace/kit /workspace/golden
  cd /workspace/golden
  find debs -name '*.deb' > /tmp/sumlist
  find bin -type f >> /tmp/sumlist
  ls fonts.tar *.tar.zst kit.tar.gz transfer.json godot-project/* godot-project-bare/* >> /tmp/sumlist 2>/dev/null || true
  xargs sha256sum < /tmp/sumlist > SHA256SUMS
  cd /workspace
  echo "SHA256SUMS over $(wc -l < /tmp/sumlist) artifacts"
}

step seed_fresh_volume seed_fresh_volume
step refresh_checkout refresh_checkout
step stage_golden stage_golden
step install_render_stack install_render_stack
step rebuild_venv rebuild_venv
step build_engine build_engine
step reextract_kbs reextract_kbs
step bundle_agents bundle_agents
step resolve_pull_refs resolve_pull_refs
step vram_table vram_table
step measure_throughput measure_throughput
always geo_recalibrate geo_recalibrate
always reconcile_tapes reconcile_tapes
always authoritative_qualgate authoritative_qualgate
always final_pack_and_sums final_pack_and_sums
echo "PREP_POD_COMPLETE — pull /workspace/qa/authoritative-probe.png + vram_table.json for the local audit"
