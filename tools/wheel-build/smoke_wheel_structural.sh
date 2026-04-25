#!/usr/bin/env bash
# Structural wheel smoke: verify llama_cpp/lib/libllama.* presence + size.
# --expect-arch ARCH also asserts the bundled binary targets that arch.
# Usage: bash smoke_wheel_structural.sh [--expect-arch ARCH]

set -euxo pipefail

expect_arch=""
while [ $# -gt 0 ]; do
  case "$1" in
    --expect-arch) expect_arch="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

python - "${expect_arch}" <<'PY'
import pathlib
import subprocess
import sys
import tempfile
import zipfile

expect_arch = sys.argv[1] or None

whls = sorted(pathlib.Path('dist').glob('lilbee-*.whl'))
assert len(whls) == 1, f'expected exactly one lilbee wheel: {whls}'
whl = whls[0]
print(f'inspecting {whl.name} ({whl.stat().st_size:,} bytes)')

if expect_arch:
    # Wheel filename platform tag must encode the target arch
    assert expect_arch in whl.name, f'wheel platform tag missing {expect_arch!r}: {whl.name}'

with zipfile.ZipFile(whl) as zf:
    libs = [n for n in zf.namelist() if n.startswith('llama_cpp/lib/')]
    print(f'llama_cpp/lib entries: {len(libs)}')
    for lib in libs:
        size = zf.getinfo(lib).file_size
        print(f'  {lib} ({size:,} bytes)')

    llama_libs = [n for n in libs if 'llama' in pathlib.Path(n).name.lower()]
    assert llama_libs, f'wheel ships no libllama.* in llama_cpp/lib/: {libs}'

    # libllama on every backend is at least 50 KB compressed -- a
    # near-empty stub would mean cmake silently produced nothing.
    for lib in llama_libs:
        size = zf.getinfo(lib).file_size
        if size < 50_000:
            raise AssertionError(f'{lib} is suspiciously small ({size} bytes)')

    if expect_arch:
        # Extract a libllama and ask `file` what it actually is. macOS
        # uses Mach-O headers ("Mach-O 64-bit dynamically linked shared
        # library x86_64") which `file` reports verbatim.
        with tempfile.TemporaryDirectory() as td:
            zf.extract(llama_libs[0], td)
            extracted = pathlib.Path(td) / llama_libs[0]
            out = subprocess.check_output(['file', str(extracted)], text=True)
            print(out.strip())
            assert expect_arch in out, f'{llama_libs[0]} arch mismatch: expected {expect_arch}, got: {out}'

    print('structural smoke OK: '
          f'{len(llama_libs)} libllama variant(s)'
          + (f' (arch={expect_arch})' if expect_arch else ''))
PY
