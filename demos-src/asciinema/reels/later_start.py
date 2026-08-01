#!/usr/bin/env python3
"""later-start: every launch after the first.

Same binary, same data, same question as first-start and cold-start -- the only variable
is that the unpack cache is already there. The reel carries on into a question and a
cited answer, because the README's point is not just that it opens quickly but that you
are working immediately.
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402
import lite  # noqa: E402

NAME = "later-start"
COLS, ROWS = 128, 41
MUST_STRINGS = ("personal encyclopedia", "what is lilbee in one sentence", "README.md")

QUESTION = "what is lilbee in one sentence?"


def record(cast: pathlib.Path) -> dict:
    root = lite.ensure()
    if not lite.UNPACK_CACHE.exists():
        # A "later" start with no cache is a first start wearing the wrong label.
        raise SystemExit("unpack cache missing; run first-start (or launch once) before this")

    s = drive.Session("reel-laterstart", COLS, ROWS, cast)
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    s.start(lite.BINARY, env={"LILBEE_DATA": str(root)})
    try:
        s.mark("boot_end")
        timings["to_chat"] = s.wait_for(r"personal encyclopedia", timeout=120)
        time.sleep(1.6)

        s.key("i", after=0.5)
        s.wait_for(r"INSERT", timeout=10)
        s.key("C-u", after=0.3)
        s.type_text(QUESTION, rate=0.045)
        time.sleep(0.8)
        s.key("Enter", after=0.8)
        timings["answer"] = s.wait_for(r"README\.md", timeout=600)
        time.sleep(3.5)

        timings["total"] = time.monotonic() - t0
        s.mark("payload_end")
        time.sleep(1.0)
    finally:
        s.kill()
    timings["marks"] = dict(s.marks)
    timings["motion_spans"] = list(s.motion_spans)
    return timings


if __name__ == "__main__":
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/later-start.cast")))
