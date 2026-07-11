#!/usr/bin/env bash
# Core-functionality gate for a built lilbee artifact (standalone executable or
# an installed wheel's entrypoint). Drives the user-facing surface end to end:
# engine inference (self-check: chat + embedding), document ingest + search,
# a RAG ask, and an http-mode crawl. Models come from a pre-populated models
# directory (the ci-models mirror in CI); nothing here contacts HuggingFace.
#
# Reads:
#   LILBEE_EXE   path to the lilbee entrypoint under test (required)
#   MODELS_DIR   models dir holding the mirror set (default: platform dir)
#   SKIP_CRAWL   set to 1 to skip the crawl leg (artifact lacks the crawler extra)

set -euxo pipefail

exe="${LILBEE_EXE:?LILBEE_EXE is required}"

case "$(uname -s)" in
  Darwin) default_models="$HOME/Library/Application Support/lilbee/models" ;;
  MINGW* | MSYS* | CYGWIN*) default_models="$HOME/AppData/Local/lilbee/models" ;;
  *) default_models="$HOME/.local/share/lilbee/models" ;;
esac
models_dir="${MODELS_DIR:-$default_models}"
export LILBEE_MODELS_DIR="${models_dir}"

chat_gguf=$(find "${models_dir}" -name 'Qwen3-0.6B-Q8_0.gguf' \( -type f -o -type l \) | head -1)
embed_gguf=$(find "${models_dir}" -name 'nomic-embed-text-v1.5.Q4_K_M.gguf' \( -type f -o -type l \) | head -1)
[ -n "${chat_gguf}" ] || { echo "chat model gguf not found under ${models_dir}" >&2; exit 1; }
[ -n "${embed_gguf}" ] || { echo "embed model gguf not found under ${models_dir}" >&2; exit 1; }

# Leg 1: engine inference through the same launch builder the fleet uses.
"${exe}" self-check --chat-model-path "${chat_gguf}" --embed-model-path "${embed_gguf}"

# Legs 2-4 share one isolated knowledge base seeded with a marker document.
work=$(mktemp -d)
data_dir="${work}/data"
export LILBEE_CHAT_MODEL="Qwen/Qwen3-0.6B-GGUF"

cat > "${work}/smoke-doc.md" <<'EOF'
# Blue quartz operations manual

The blue quartz resonator array is calibrated to 432 megahertz and must be
kept below 40 degrees. The calibration engineer on record is Ada Marlowe.
EOF

# Leg 2: ingest + search. The featured embedder is the mirrored nomic model,
# so ingest resolves it locally.
"${exe}" --data-dir "${data_dir}" add "${work}/smoke-doc.md"
search_out=$("${exe}" --data-dir "${data_dir}" search "blue quartz resonator")
echo "${search_out}"
echo "${search_out}" | grep -qi "quartz" || { echo "FAIL: search returned no quartz hit" >&2; exit 1; }

# Leg 3: RAG ask through the real chat engine. Asserts the pipeline produces
# an answer, not the answer's content: a 0.6B model quoting the document is
# not deterministic, and retrieval correctness is already covered by leg 2.
ask_out=$("${exe}" --data-dir "${data_dir}" ask "What frequency is the blue quartz resonator calibrated to?")
echo "${ask_out}"
[ -n "$(echo "${ask_out}" | tr -d '[:space:]')" ] || { echo "FAIL: ask returned an empty answer" >&2; exit 1; }
case "${ask_out}" in *"Error:"*) echo "FAIL: ask surfaced an error" >&2; exit 1 ;; esac

# Leg 4: crawl a URL in the default http render mode (no browser needed) and
# confirm the page text became searchable.
if [ "${SKIP_CRAWL:-0}" != "1" ]; then
  "${exe}" --data-dir "${data_dir}" add "https://example.com"
  crawl_out=$("${exe}" --data-dir "${data_dir}" search "illustrative examples in documents")
  echo "${crawl_out}"
  echo "${crawl_out}" | grep -qi "example" || { echo "FAIL: crawled page not searchable" >&2; exit 1; }
fi

echo "ARTIFACT SMOKE PASSED: self-check, ingest, search, ask${SKIP_CRAWL:+ (crawl skipped)}"
