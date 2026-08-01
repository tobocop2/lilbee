#!/usr/bin/env python3
"""tui-multi-gpu-self-index: lilbee indexing its own source across two cards.

The corpus is a slice of lilbee's own source -- the retrieval and core packages -- which
makes the reel self-referential on purpose: the thing being indexed is the thing doing the
indexing, so nobody has to take a staged corpus on trust. A slice rather than the whole
package because the full tree took over forty minutes to embed, which is not a reel. The placement drawer stays open through the ingest, so both cards are visible
doing work rather than one card doing work and another sitting idle in a table.

Recorded on a pod. On one card this reel has no subject.
"""
from __future__ import annotations

import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "tui-multi-gpu-self-index"
COLS, ROWS = 128, 41
MUST_STRINGS = ("Placement", "Background Tasks", "Sources:")
TAIL_FORBID = ("Cancel stream",)
SPEED_WINDOWS = ("ingest", "gen")

ROOT = "/workspace/reelroot"
CORPUS = "/workspace/corpus-small"
QUESTION = "how does lilbee decide which GPU a model runs on?"


def record(cast: pathlib.Path) -> dict:
    s = drive.Session("reel-selfindex", COLS, ROWS, cast)
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    s.start("lilbee", env={"LILBEE_DATA": ROOT})
    try:
        timings["boot"] = s.wait_for(r"personal encyclopedia", timeout=300)
        time.sleep(1.5)
        s.repaint()
        s.wait_for(r"personal encyclopedia", timeout=30)
        time.sleep(0.6)
        s.mark("boot_end")

        # 1. Both cards, before any work.
        s.key("C-g", after=1.5)
        s.wait_for(r"Placement", timeout=40)
        time.sleep(2.4)

        # 2. Point it at its own source.
        s.key("i", after=0.6)
        s.key("C-u", after=0.3)
        s.type_text(f"/add {CORPUS}", rate=0.05)
        time.sleep(1.0)
        s.key("Enter", after=1.2)
        time.sleep(2.5)

        # 3. The Task Center, with the placement drawer still up beside it.
        s.esc()
        time.sleep(0.4)
        s.key("t", after=0.9)
        s.wait_for(r"Background Tasks", timeout=40)
        s.mark("ingest_start")
        # Fail fast on a failed add instead of waiting out the timeout. lilbee marks the
        # whole add "failed" when any single file yields no text, so two empty __init__.py
        # files were enough to leave this reel waiting forty minutes for a "done" that was
        # never coming.
        deadline = time.monotonic() + 1500
        while time.monotonic() < deadline:
            screen = s.screen()
            if re.search(r"add\s+(done|complete)|all caught up", screen):
                break
            if re.search(r"add\s+failed", screen):
                raise SystemExit(f"ingest failed on camera:\n{screen}")
            time.sleep(1.0)
        timings["ingest"] = 0.0
        s.mark("ingest_end")
        time.sleep(2.4)

        # 4. Ask the source about itself.
        s.goto("Chat", forward=False, limit=8, marker=r"Slash commands")
        time.sleep(0.8)
        s.ask(QUESTION, rate=0.05)
        s.mark("gen_start")
        timings["answer"] = s.await_answer()
        s.mark("gen_end")
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
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                              else "/tmp/tui-multi-gpu-self-index.cast")))
