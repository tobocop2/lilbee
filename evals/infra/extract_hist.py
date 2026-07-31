"""Exact extraction percentiles over every traced file, updated incrementally.

elapsed_ms is a small integer, so a bucket-per-millisecond histogram is exact and
costs O(new lines) per tick rather than O(all lines). State is a byte offset per
trace file plus the counts, so a tick reads only what was appended since the last
one and the percentiles still cover the whole corpus.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

PROF = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/root/prof")
STATE = PROF / ".hist.json"
CAP_MS = 60_000
PATTERN = re.compile(rb"elapsed_ms=(\d+)")

state = json.loads(STATE.read_text()) if STATE.exists() else {"offsets": {}, "counts": {}}
offsets: dict[str, int] = state["offsets"]
counts: dict[str, int] = state["counts"]

for path in sorted(PROF.glob("*.trace.log")):
    name = path.name
    start = offsets.get(name, 0)
    size = path.stat().st_size
    if size < start:          # rotated or truncated: start over for this file
        start = 0
    if size == start:
        continue
    with path.open("rb") as fh:
        fh.seek(start)
        chunk = fh.read(size - start)
    tail = chunk.rfind(b"\n")  # never parse a half-written final line
    if tail == -1:
        continue
    for match in PATTERN.finditer(chunk[: tail + 1]):
        ms = min(int(match.group(1)), CAP_MS)
        key = str(ms)
        counts[key] = counts.get(key, 0) + 1
    offsets[name] = start + tail + 1

STATE.write_text(json.dumps({"offsets": offsets, "counts": counts}))

total = sum(counts.values())
if not total:
    print("  (no traces yet)")
    raise SystemExit

ordered = sorted((int(k), v) for k, v in counts.items())


def at(p: float) -> int:
    """Smallest millisecond value at or below which *p* of samples fall."""
    target = p * total
    seen = 0
    for ms, n in ordered:
        seen += n
        if seen >= target:
            return ms
    return ordered[-1][0]


mean = sum(ms * n for ms, n in ordered) / total
print(f"  files {total:,}   mean {mean:.2f} ms")
print(f"  p50 {at(.50)}   p90 {at(.90)}   p99 {at(.99)}   p99.9 {at(.999)}   max {ordered[-1][0]} ms")
under3 = sum(n for ms, n in ordered if ms <= 3)
print(f"  {100 * under3 / total:.1f}% at or under 3 ms")
