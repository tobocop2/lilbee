#!/usr/bin/env bash
# Canonical lilbee + engine provisioning for pods. NO COMPILATION.
# Installs the released lilbee and its prebuilt, self-contained engine
# (llama-server + ggml/llama/mtmd libs + llama-swap + gguf-parser) from the
# per-backend PEP 503 index. Source this; do NOT set LILBEE_LLAMA_SERVER_PATH.
#
#   BACKEND=cu124 VENV=/root/venv source provision_engine.sh          # reels (release)
#   BACKEND=cu124 REF=fix/embed-429-cold-start-retry source provision_engine.sh  # test a branch
#
# BACKEND: cu121 | cu124 | cu125 | cpu | compat  (default cu124 for NVIDIA pods)
# REF    : optional git ref whose python is overlaid on top of the release engine.
set -x
: "${BACKEND:=cu124}"
: "${VENV:=/root/venv}"
: "${REPO:=https://github.com/tobocop2/lilbee}"

python3 -m venv "$VENV" && "$VENV/bin/pip" install -q --upgrade pip >/dev/null 2>&1
# Install the engine FIRST and with retries. It is a ~905MB wheel and the index
# can be slow on some pods (2026-07-16: a pod self-terminated on its linger while
# this was still downloading). `pip install --pre lilbee` alone does NOT reliably
# pull lilbee-engine as a dependency from the extra index, so name it explicitly.
for attempt in 1 2 3; do
  "$VENV/bin/pip" install --pre --retries 5 --timeout 120 lilbee-engine \
    --extra-index-url "https://lilbee.sh/${BACKEND}/" && break
  echo "engine install attempt $attempt failed; retrying" >&2; sleep 10
done
"$VENV/bin/pip" install -q --pre lilbee --extra-index-url "https://lilbee.sh/${BACKEND}/"
# Optional: overlay an unmerged branch's python while keeping the prebuilt engine + deps.
if [ -n "${REF:-}" ]; then
  "$VENV/bin/pip" install -q --no-deps --force-reinstall "git+${REPO}@${REF}"
fi
export PATH="$VENV/bin:$PATH"

# Sanity: bundled engine binary must be present (proves no stub / no compile needed).
ENGINE=$("$VENV/bin/python" -c "import lilbee_engine,glob,pathlib; b=pathlib.Path(lilbee_engine.__file__).parent/'bin'; print(next(iter(glob.glob(str(b/'llama-server'))),''))" 2>/dev/null)
if [ -z "$ENGINE" ] || ! command -v lilbee >/dev/null; then
  echo "PROVISION_FAILED: lilbee=$(command -v lilbee) engine=$ENGINE (backend=$BACKEND ref=${REF:-none})" >&2
  return 1 2>/dev/null || exit 1
fi
echo "engine OK: $(lilbee --version 2>/dev/null | head -1) | $ENGINE | backend=$BACKEND ref=${REF:-release}"
