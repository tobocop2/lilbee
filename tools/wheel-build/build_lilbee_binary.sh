#!/usr/bin/env bash
# Build the lilbee single-file binary with Nuitka.
#
# Args via env:
#   ASSET_NAME       output filename, e.g. lilbee-macos-arm64
#   PRODUCT_VERSION  Nuitka --product-version (int-tuple, e.g. 0.6.66.456)
#
# Run from the repo root after `uv sync --extra release` and
# `uv pip install 'nuitka[onefile]>=4.0,<5'`.

set -euxo pipefail

ASSET_NAME="${ASSET_NAME:?ASSET_NAME is required}"
PRODUCT_VERSION="${PRODUCT_VERSION:?PRODUCT_VERSION is required (int-tuple, e.g. 0.6.66.456)}"

# python-multipart's multipart/ shim shadows the real multipart package; remove
# it so litestar gets the real module. Idempotent.
uv pip uninstall python-multipart >/dev/null 2>&1 || true

# Bundle en_core_web_sm so concept extraction works in the frozen binary.
# It is loaded by name (spacy.load), never imported, so the Nuitka run below
# pulls in the package, its model data, and its distribution metadata.
# uv-managed Python ships without pip; install the matching wheel directly.
if ! uv run --no-sync python -c "import en_core_web_sm" >/dev/null 2>&1; then
    SPACY_VER=$(uv run --no-sync python -c "import spacy; print(spacy.__version__)")
    case "$SPACY_VER" in
        3.7.*) MODEL_VER="3.7.1" ;;
        3.8.*) MODEL_VER="3.8.0" ;;
        *)     MODEL_VER="3.8.0" ;;
    esac
    uv pip install \
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-${MODEL_VER}/en_core_web_sm-${MODEL_VER}-py3-none-any.whl"
fi

# Top-level *__mypyc*.so helpers (chardet, charset_normalizer, mypy) live at
# site-packages root, not inside the package dir, so --include-package-data
# misses them. Glob them explicitly; forward-compatible across version bumps.
SITE_PKG=$(uv run --no-sync python -c "import site; print(site.getsitepackages()[0])")
MYPYC_FLAGS=()
for f in "$SITE_PKG"/*__mypyc*.so; do
    [ -e "$f" ] || continue
    MYPYC_FLAGS+=("--include-data-files=$f=$(basename "$f")")
done

# Bundle the local inference engine when its package is installed. The release
# build installs packaging/engine-wheel after build_llama_server.sh fills its
# bin/ with the self-contained llama-server (binary + ggml/llama/mtmd libs with a
# baked rpath) plus the llama-swap and gguf-parser helpers, so
# --include-package-data ships the whole engine inside the onefile and the runtime
# resolver finds each via lilbee_engine.get_*_path(). Optional so a build without
# the engine wheel still succeeds (resolver falls back to PATH).
LLAMA_SERVER_FLAGS=()
if uv run --no-sync python -c "import lilbee_engine" >/dev/null 2>&1; then
    # --include-package pulls the __init__.py and, on Linux, the .so libraries
    # (Nuitka classifies a bare .so as an extension module and places it beside
    # the binary). --include-package-data ships the executables (llama-server,
    # llama-swap, gguf-parser) as data.
    LLAMA_SERVER_FLAGS+=(--include-package=lilbee_engine)
    LLAMA_SERVER_FLAGS+=(--include-package-data=lilbee_engine)
    # Platform gap: macOS .dylib and Windows .dll are NOT extension modules, so
    # --include-package-data routes them through Nuitka's DLL tracker, which
    # discards the engine's library closure (nothing Nuitka scans references the
    # bundled llama-server) and ships a server that cannot start. Force those in
    # verbatim with --include-data-files (rpath/exec bit preserved). Linux .so is
    # deliberately excluded: it is already an extension, and data-files-ing it
    # FATALs with an extension/data conflict.
    ENGINE_BIN=$(uv run --no-sync python -c "import lilbee_engine, pathlib; print(pathlib.Path(lilbee_engine.__file__).parent / 'bin')")
    for _f in "${ENGINE_BIN}"/*.dylib "${ENGINE_BIN}"/*.dll; do
        [ -f "${_f}" ] || continue
        LLAMA_SERVER_FLAGS+=(--include-data-files="${_f}=lilbee_engine/bin/$(basename "${_f}")")
    done
fi

# chardet ships its detection models as .bin data files that crawl4ai loads on
# every http fetch. Neither --include-package-data nor --include-data-dir keeps
# .bin (Nuitka does not classify it as data), so the frozen crawl path dies with
# a missing models.bin. Force each .bin in explicitly. The package's .py/.so
# come from --include-package=chardet above, so only the .bin data is listed.
CHARDET_FLAGS=()
CHARDET_MODELS=$(uv run --no-sync python -c "import chardet, pathlib; d = pathlib.Path(chardet.__file__).parent / 'models'; print(d if d.is_dir() else '')" 2>/dev/null || true)
if [ -n "${CHARDET_MODELS}" ]; then
    for _f in "${CHARDET_MODELS}"/*.bin; do
        [ -f "${_f}" ] || continue
        CHARDET_FLAGS+=(--include-data-files="${_f}=chardet/models/$(basename "${_f}")")
    done
fi
# Spawned via `python -m`, never statically imported, so Nuitka's import
# following misses it; include it explicitly or the splash subprocess dies.
SPLASH_FLAGS=(--include-module=lilbee.runtime._splash_runner)

uv run --no-sync python -m nuitka \
    --mode=onefile \
    --user-plugin=tools/wheel-build/playwright_node_verbatim.py \
    --no-deployment-flag=self-execution \
    --onefile-cache-mode=cached \
    --onefile-tempdir-spec='{CACHE_DIR}/lilbee/{VERSION}' \
    --product-name=lilbee \
    --product-version="$PRODUCT_VERSION" \
    --output-filename="$ASSET_NAME" \
    --output-dir=dist \
    --assume-yes-for-downloads \
    --nofollow-import-to=*.tests.* \
    --nofollow-import-to=tkinter --nofollow-import-to=_tkinter \
    --include-package=lancedb            --include-package-data=lancedb \
    --include-package=tree_sitter_language_pack --include-package-data=tree_sitter_language_pack \
    --include-package=tiktoken           --include-package-data=tiktoken \
    --include-package=tiktoken_ext       --include-package-data=tiktoken_ext \
    --include-package-data=numpy \
    --include-package=kreuzberg          --include-package-data=kreuzberg \
    --include-package=litellm            --include-package=litellm.llms      --include-package-data=litellm \
    --include-package=crawl4ai           --include-package-data=crawl4ai \
    --include-package=fake_useragent     --include-package-data=fake_useragent \
    --include-package=chardet            --include-package-data=chardet \
    --include-package=playwright \
    --enable-plugin=spacy \
    --include-package=spacy              --include-package-data=spacy \
    --include-package=en_core_web_sm     --include-package-data=en_core_web_sm \
    --spacy-language-model=en_core_web_sm \
    --include-package=graspologic_native --include-package-data=graspologic_native \
    --include-package=textual            --include-package-data=textual \
    --include-package=rich               --include-package-data=rich \
    --include-package=litestar           --include-package-data=litestar \
    --include-package=mcp                --include-package-data=mcp \
    --include-distribution-metadata=lilbee \
    --include-distribution-metadata=litellm \
    --include-distribution-metadata=Crawl4AI \
    --include-distribution-metadata=spacy \
    --include-distribution-metadata=en_core_web_sm \
    --include-distribution-metadata=catalogue \
    --include-data-dir=src/lilbee/cli/tui=lilbee/cli/tui \
    --include-data-dir=src/lilbee/skills=lilbee/skills \
    --include-data-files=src/lilbee/featured_models.toml=lilbee/featured_models.toml \
    "${SPLASH_FLAGS[@]}" \
    "${MYPYC_FLAGS[@]}" \
    "${CHARDET_FLAGS[@]}" \
    "${LLAMA_SERVER_FLAGS[@]}" \
    src/lilbee/__main__.py
