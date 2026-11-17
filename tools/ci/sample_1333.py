"""Temporary CI sampler: log machine load and memory while the tests run."""

from __future__ import annotations

import os
import sys
import time

import psutil

INTERVAL_S = 1.0
LATE_S = 3.0
EVERY = 5


def line(late: float) -> str:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    pythons = []
    for p in psutil.process_iter(["name", "memory_info", "cpu_percent", "num_threads"]):
        info = p.info
        if info["memory_info"] is None:
            continue
        pythons.append((info["memory_info"].rss, info["cpu_percent"] or 0.0, info["name"]))
    pythons.sort(reverse=True)
    top = " ".join(f"{n}:{r // 2**20}M/{c:.0f}%" for r, c, n in pythons[:5])
    flag = "LATE" if late > LATE_S else "ok"
    return (
        f"SAMPLE {flag} t={time.strftime('%H:%M:%S')} late={late:.1f}s "
        f"cpu={psutil.cpu_percent(None):.0f}% load={os.getloadavg()[0]:.1f} "
        f"avail={vm.available // 2**20}M swap={sw.used // 2**20}M "
        f"procs={len(pythons)} top={top}"
    )


def main() -> None:
    psutil.cpu_percent(None)
    last = time.monotonic()
    count = 0
    while True:
        time.sleep(INTERVAL_S)
        now = time.monotonic()
        late = now - last - INTERVAL_S
        last = now
        count += 1
        if late > LATE_S or count % EVERY == 0:
            print(line(late), flush=True)


if __name__ == "__main__":
    sys.exit(main())
