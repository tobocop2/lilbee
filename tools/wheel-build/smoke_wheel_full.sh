#!/usr/bin/env bash
# Install a lilbee wheel and run --version, --help, and --json self-check.
# Usage: bash smoke_wheel_full.sh <wheel-glob>

set -euxo pipefail

wheel_glob="${1:?wheel glob required (e.g. 'dist/*.whl')}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pre-seed an old typer to verify the wheel's >=0.12 pin actually upgrades it.
pip install 'typer==0.9.4'
# shellcheck disable=SC2086 -- wheel_glob must expand
pip install --pre ${wheel_glob}

python -c "
import typer
v = tuple(int(p) for p in typer.__version__.split('.')[:2])
assert v >= (0, 12), f'typer={typer.__version__} < 0.12'
print(f'typer {typer.__version__} OK')
"

python -c "
import lilbee
from lilbee.cli import app
from lilbee import code_chunker, ingest, embedder, store, query, config, chunk
import llama_cpp
print('llama_cpp loaded from:', llama_cpp.__file__)
print('lilbee imports OK')
"

expected=$(python -c "import tomllib, pathlib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])")
stdout_file=$(mktemp)
stderr_file=$(mktemp)
lilbee --version >"$stdout_file" 2>"$stderr_file"
test "$(cat "$stdout_file")" = "lilbee ${expected}"
if grep -qE "No such option|Type not yet supported|RuntimeError" "$stderr_file"; then
  cat "$stderr_file"; exit 1
fi
lilbee --help >/dev/null 2>"$stderr_file"
if grep -qE "No such option|Type not yet supported|RuntimeError" "$stderr_file"; then
  cat "$stderr_file"; exit 1
fi

export HF_HUB_DISABLE_PROGRESS_BARS=1
out_file=$(mktemp)
rc=0
lilbee --json self-check > "$out_file" 2>&1 || rc=$?
echo "=== self-check (exit=${rc}) ==="
cat "$out_file"
[ "${rc}" = "0" ]
python "${script_dir}/parse_self_check.py" "$out_file"
