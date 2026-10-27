#!/usr/bin/env bash
# Core-functionality gate for a built lilbee artifact (standalone executable or
# an installed wheel's entrypoint). Drives the user-facing surface end to end:
# bundled extras (self-check-extras), engine inference
# (self-check: chat + embedding), document ingest + search, a RAG ask, and an
# http-mode crawl, and a browser-mode crawl that proves the frozen binary loads
# the crawler's native library and starts Chromium. Models come from a pre-populated models
# directory (the ci-models mirror in CI); nothing here contacts HuggingFace.
#
# Reads:
#   LILBEE_EXE   path to the lilbee entrypoint under test (required)
#   MODELS_DIR   models dir holding the mirror set (default: platform dir)
#   SKIP_CRAWL   set to 1 to skip the crawl legs (artifact lacks the crawler extra)
#   BROWSER_TIMEOUT  seconds the browser-mode crawl may run, including the
#                    first-use Chromium download (default: 900)

set -euxo pipefail

exe="${LILBEE_EXE:?LILBEE_EXE is required}"

# Run a command under a watchdog when timeout(1) is missing (macOS runners).
# Job control gives the command and the watchdog a process group each, so both
# paths kill whole groups: a timeout takes the command and its children, and a
# finish takes the watchdog and its sleep. Exits 124 on a timeout.
watchdog_timeout() {  # seconds command...
  local secs="$1" cmd_pid watchdog_pid rc=0
  shift
  set -m
  "$@" &
  cmd_pid=$!
  ( sleep "${secs}"; kill -TERM -- "-${cmd_pid}" 2> /dev/null ) &
  watchdog_pid=$!
  set +m
  wait "${cmd_pid}" || rc=$?
  kill -TERM -- "-${watchdog_pid}" 2> /dev/null || true
  wait "${watchdog_pid}" 2> /dev/null || true
  [ "${rc}" -le 128 ] || rc=124
  return "${rc}"
}

# Run a command under a hard time limit.
with_timeout() {  # seconds command...
  if command -v timeout > /dev/null; then
    timeout "$@"
  elif command -v gtimeout > /dev/null; then
    gtimeout "$@"
  else
    watchdog_timeout "$@"
  fi
}

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

# Leg 0: bundled extras import.
# Skipped with the crawl leg: the check needs the crawler extra.
if [ "${SKIP_CRAWL:-0}" != "1" ]; then
  "${exe}" self-check-extras
fi

# Leg 1: engine inference through the same launch builder the fleet uses.
"${exe}" self-check --chat-model-path "${chat_gguf}" --embed-model-path "${embed_gguf}"

# Legs 2-4 share one isolated knowledge base seeded with a marker document.
# Both model roles default to unconfigured, so the smoke names its refs
# explicitly; they resolve against the mirrored files found above.
work=$(mktemp -d)
data_dir="${work}/data"
export LILBEE_CHAT_MODEL="Qwen/Qwen3-0.6B-GGUF"
export LILBEE_EMBEDDING_MODEL="nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"

cat > "${work}/smoke-doc.md" <<'EOF'
# Blue quartz operations manual

The blue quartz resonator array is calibrated to 432 megahertz and must be
kept below 40 degrees. The calibration engineer on record is Ada Marlowe.
EOF

# Leg 2: ingest + search.
# Every captured leg records its exit code and prints what it captured BEFORE
# it asserts. A leg that fails inside a command substitution under `set -e`
# ends the script at the assignment, and the error the CLI printed is lost.
"${exe}" --data-dir "${data_dir}" add "${work}/smoke-doc.md"
rc=0
search_out=$("${exe}" --data-dir "${data_dir}" search "blue quartz resonator") || rc=$?
echo "${search_out}"
[ "${rc}" -eq 0 ] || { echo "FAIL: search exited ${rc}" >&2; exit 1; }
echo "${search_out}" | grep -qi "quartz" || { echo "FAIL: search returned no quartz hit" >&2; exit 1; }

# Leg 3: RAG ask through the real chat engine. Asserts the pipeline produces
# an answer, not the answer's content: a 0.6B model quoting the document is
# not deterministic, and retrieval correctness is already covered by leg 2.
rc=0
ask_out=$("${exe}" --data-dir "${data_dir}" ask "What frequency is the blue quartz resonator calibrated to?") || rc=$?
echo "${ask_out}"
[ "${rc}" -eq 0 ] || { echo "FAIL: ask exited ${rc}" >&2; exit 1; }
[ -n "$(echo "${ask_out}" | tr -d '[:space:]')" ] || { echo "FAIL: ask returned an empty answer" >&2; exit 1; }
case "${ask_out}" in *"Error:"*) echo "FAIL: ask surfaced an error" >&2; exit 1 ;; esac

# Leg 4: crawl a URL in the default http render mode (no browser needed) and
# confirm the page text became searchable.
if [ "${SKIP_CRAWL:-0}" != "1" ]; then
  "${exe}" --data-dir "${data_dir}" add "https://example.com"
  rc=0
  crawl_out=$("${exe}" --data-dir "${data_dir}" search "domain is for use in examples") || rc=$?
  echo "${crawl_out}"
  [ "${rc}" -eq 0 ] || { echo "FAIL: crawl search exited ${rc}" >&2; exit 1; }
  echo "${crawl_out}" | grep -qi "domain is for use in" || { echo "FAIL: crawled page not searchable" >&2; exit 1; }
fi

# Leg 5: crawl one page in browser mode into a fresh knowledge base and a fresh
# Chromium cache. The first crawl downloads Chromium, so the binary must load
# the crawler's native library, fetch the browser, launch it and render the
# page. The cache starts empty, so the headless shell can only be there if this
# crawl installed it: a crawl that ignored the setting and fetched over HTTP
# leaves the cache empty and fails. The hard timeout keeps a hung browser from
# holding the runner.
browser=""
if [ "${SKIP_CRAWL:-0}" != "1" ]; then
  browser=", browser crawl"
  browser_data="${work}/browser-data"
  browsers_dir="${work}/browsers"
  mkdir -p "${browsers_dir}"
  # A native Windows executable needs a Windows path in the environment.
  if command -v cygpath > /dev/null; then
    browsers_dir=$(cygpath -m "${browsers_dir}")
  fi
  rc=0
  LILBEE_CRAWL_RENDER_MODE=browser PLAYWRIGHT_BROWSERS_PATH="${browsers_dir}" \
    with_timeout "${BROWSER_TIMEOUT:-900}" \
    "${exe}" --data-dir "${browser_data}" add "https://example.com" || rc=$?
  [ "${rc}" -eq 0 ] || { echo "FAIL: browser crawl exited ${rc}" >&2; exit 1; }
  compgen -G "${browsers_dir}/chromium_headless_shell-*/INSTALLATION_COMPLETE" > /dev/null \
    || { echo "FAIL: the browser crawl did not install or launch Chromium" >&2; exit 1; }
  rc=0
  browser_out=$("${exe}" --data-dir "${browser_data}" search "domain is for use in documentation") || rc=$?
  echo "${browser_out}"
  [ "${rc}" -eq 0 ] || { echo "FAIL: browser crawl search exited ${rc}" >&2; exit 1; }
  # Match page body text; the printed URL alone must not satisfy the check.
  echo "${browser_out}" | grep -qi "domain is for use in" || { echo "FAIL: browser-crawled page not searchable" >&2; exit 1; }
fi

skipped=""
[ "${SKIP_CRAWL:-0}" = "1" ] && skipped=" (extras and crawl skipped)"
echo "ARTIFACT SMOKE PASSED: extras, self-check, ingest, search, ask${browser}${skipped}"
