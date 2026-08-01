#!/usr/bin/env bash
# Re-verify lilbee's page-dataset export at full corpus scale after the streaming
# and columnar-join changes (bb-qlc9h, PR #646).
#
# Measures rather than asserts: peak memory is sampled for the whole run and
# sliced per format afterwards, so "bounded by one batch" is falsifiable instead
# of assumed. Nothing here fixes anything; a failure is reported, not repaired.
set -uo pipefail
OUT=/workspace/exportcheck
COMMIT="${COMMIT:-c98c0fadc}"
rm -rf "$OUT"; mkdir -p "$OUT"
exec > "$OUT/run.log" 2>&1
say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
mark() { printf '%s %s\n' "$(date -u +%s)" "$1" >> "$OUT/phases"; }

# MemAvailable every 5s for the whole run. Sliced by phase timestamps later, so
# each format gets its own peak rather than sharing one number.
( while :; do
    printf '%s,%s\n' "$(date -u +%s)" \
      "$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)"
    sleep 5
  done ) > "$OUT/mem.csv" 2>/dev/null &
MEMPID=$!
trap 'kill "$MEMPID" 2>/dev/null' EXIT

say "=== checkout ==="
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
rm -rf /root/src
git clone -q https://github.com/tobocop2/lilbee /root/src || { say "FATAL clone"; exit 1; }
cd /root/src
git fetch -q origin fix/export-64bit-offsets-and-corpus-progress
git checkout -q "$COMMIT" || { say "FATAL checkout $COMMIT"; exit 1; }
HEAD_SHA=$(git rev-parse HEAD)
say "git rev-parse HEAD = $HEAD_SHA"
say "describes: $(git log --oneline -1)"
echo "$HEAD_SHA" > "$OUT/head_sha"

say "=== install ==="
uv venv --seed --python 3.12 /root/venvchk >/dev/null 2>&1
VPY=/root/venvchk/bin/python
uv pip install -q --python "$VPY" --prerelease=allow /root/src || { say "FATAL install"; exit 1; }
"$VPY" - <<'PY' | tee "$OUT/versions"
import importlib.metadata as md
for pkg in ("lilbee", "pyarrow", "lancedb", "pandas"):
    try:
        print(f"{pkg}=={md.version(pkg)}")
    except Exception as exc:
        print(f"{pkg}==ABSENT ({exc})")
PY

export LILBEE_DATA=/workspace/kb/.lilbee

say "=== index shape before exporting ==="
# Does the real index actually carry duplicate source rows? That decides whether
# the join's dedup runs in practice or is purely defensive.
"$VPY" - <<'PY' | tee "$OUT/index_shape"
import lancedb
db = lancedb.connect("/workspace/kb/.lilbee/data/lancedb")
src = db.open_table("_sources")
total = src.count_rows()
# head(count_rows()) rather than to_lance(): to_lance needs pylance, which is
# not a lilbee dependency, and the import error costs a run to discover.
names = src.head(total).column("filename").to_pylist()
print(f"_sources count(*)          = {total}")
print(f"_sources count(DISTINCT)   = {len(set(names))}")
print(f"duplicate source rows      = {total - len(set(names))}")
print(f"chunks count(*)            = {db.open_table('chunks').count_rows()}")
print(f"_page_texts count(*)       = {db.open_table('_page_texts').count_rows()}")
PY

for fmt in parquet jsonl; do
  say "=== export $fmt ==="
  mark "${fmt}_start"
  t0=$(date -u +%s)
  "$VPY" -m lilbee export "$OUT/msmarco-passages.$fmt" 2>&1 | tail -3 \
    || /root/venvchk/bin/lilbee export "$OUT/msmarco-passages.$fmt" 2>&1 | tail -3
  rc=$?
  t1=$(date -u +%s)
  mark "${fmt}_end"
  say "$fmt rc=$rc wall=$((t1 - t0))s size=$(stat -c %s "$OUT/msmarco-passages.$fmt" 2>/dev/null || echo 0) bytes"
done

say "=== verify parquet ==="
"$VPY" - <<'PY' | tee "$OUT/verify_parquet"
import pyarrow.parquet as pq

path = "/workspace/exportcheck/msmarco-passages.parquet"
pf = pq.ParquetFile(path)
rows = pf.metadata.num_rows
print(f"row count                  = {rows}")
print(f"EXPECTED                   = 8841823")
print(f"ROW COUNT VERDICT          = {'PASS' if rows == 8841823 else 'FAIL'}")
print("schema:")
for f in pf.schema_arrow:
    print(f"   {f.name:14s} {f.type}")
wide = [f.name for f in pf.schema_arrow if str(f.type) == "string"]
print(f"NARROW string columns      = {wide or 'none (all large_string)'}")

# Metadata must land on every page of its source, not just the first.
t = pq.read_table(path, columns=["source", "title", "authors", "created_at"])
import collections
n = t.num_rows
nulls = {c: t.column(c).null_count for c in ("title", "authors", "created_at")}
print(f"rows read back             = {n}")
print(f"null counts                = {nulls}")
srcs = t.column("source").to_pylist()
print(f"distinct sources           = {len(set(srcs))}")
print(f"DUPLICATION VERDICT        = {'PASS' if n == len(set(srcs)) == 8841823 else 'FAIL'}")
PY

say "=== import round-trip (parquet) ==="
"$VPY" - <<'PY' | tee "$OUT/roundtrip"
import pathlib
from lilbee.data.export import DatasetFormat, load_page_dataset

path = pathlib.Path("/workspace/exportcheck/msmarco-passages.parquet")
rows = load_page_dataset(path, DatasetFormat.PARQUET)
print(f"imported records           = {len(rows)}")
print(f"ROUND TRIP VERDICT         = {'PASS' if len(rows) == 8841823 else 'FAIL'}")
r = rows[0]
print(f"sample record              = {dict(r) if isinstance(r, dict) else r}")
PY

say "=== peak memory per phase ==="
"$VPY" - <<'PY' | tee "$OUT/memory"
base = {}
for line in open("/workspace/exportcheck/phases"):
    ts, name = line.split()
    base[name] = int(ts)
mem = []
for line in open("/workspace/exportcheck/mem.csv"):
    a, b = line.strip().split(",")
    mem.append((int(a), int(b)))
total = max(b for _, b in mem)
print(f"MemAvailable high-water (idle baseline) = {total} MB")
for fmt in ("parquet", "jsonl"):
    s, e = base.get(f"{fmt}_start"), base.get(f"{fmt}_end")
    if not s or not e:
        print(f"{fmt}: no window recorded")
        continue
    win = [b for t, b in mem if s <= t <= e]
    if not win:
        print(f"{fmt}: no samples in window")
        continue
    print(f"{fmt:8s} window {e - s:5d}s  MemAvailable min {min(win)} MB  "
          f"=> peak used above idle {(total - min(win)) / 1024:.1f} GB")
PY

say "=== DONE ==="
touch "$OUT/CHECK_DONE"
