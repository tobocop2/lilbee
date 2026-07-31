"""Sample the merged index and every shard's `_sources` row count.

The parent reports one aggregate progress bar and keeps the per-worker counters
in memory, so per-worker progress has to be read off the shards themselves.
`count_rows` is Lance metadata rather than a scan, so a tick is cheap enough to
run every 20s for six hours.

Output is one CSV line per tick: ts,merged,w0,w1,...  Writes are appended and
flushed per line so a reader tailing the file always sees whole rows.

Usage: rows_sampler.py <data_root> <out.csv|-> <interval_s>   (0 = one tick)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

DATA = Path(sys.argv[1])
OUT = sys.argv[2]
INTERVAL = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0


def count(root: Path) -> int:
    """`_sources` rows under *root*, or 0 while the store does not exist yet.

    The directory is checked before lancedb is pointed at it: `connect` CREATES
    what it is given, and a stray `shards/w*` directory from an unexpanded glob
    once inflated a worker count to 9 on an 8-GPU box.
    """
    db_dir = root / "data" / "lancedb"
    if not db_dir.is_dir():
        return 0
    try:
        import lancedb

        db = lancedb.connect(db_dir)
        names = db.table_names()
        if "_sources" not in list(getattr(names, "tables", names)):
            return 0
        return int(db.open_table("_sources").count_rows())
    except Exception:
        return 0


def shards() -> list[Path]:
    """Worker data roots, in worker order."""
    root = DATA / "shards"
    if not root.is_dir():
        return []
    found = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("w")]
    return sorted(found, key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0)


def tick() -> str:
    per = [count(p) for p in shards()]
    return ",".join([str(int(time.time())), str(count(DATA)), *(str(n) for n in per)])


def main() -> int:
    handle = sys.stdout if OUT == "-" else open(OUT, "a", buffering=1)  # noqa: SIM115
    try:
        while True:
            print(tick(), file=handle, flush=True)
            if INTERVAL <= 0:
                return 0
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        return 0
    finally:
        if handle is not sys.stdout:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
