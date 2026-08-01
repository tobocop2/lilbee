#!/usr/bin/env python3
"""tui-manual-placement: override where each model runs, on a two-card machine.

Recorded on a pod because there is nothing to place on one card. The placement screen
shows a row per role and a column per GPU, with -/+ for replicas, and the reel moves chat
and embedding onto different cards, previews the plan, and applies it.

Bindings come from the screen's own footer: ctrl+r previews, ctrl+s applies, ctrl+x
returns to automatic placement. The reel ends back on Auto so it does not leave the
viewer thinking manual placement is something you have to do.
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "tui-manual-placement"
COLS, ROWS = 128, 41
MUST_STRINGS = ("Placement", "Preview", "Apply", "replicas")
SPEED_WINDOWS = ()

ROOT = "/workspace/reelroot"


def record(cast: pathlib.Path) -> dict:
    s = drive.Session("reel-placement", COLS, ROWS, cast)
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
        s.esc(2)
        time.sleep(0.5)

        # 1. What automatic placement decided, with both cards listed.
        s.key("C-g", after=1.5)
        s.wait_for(r"Placement", timeout=40)
        time.sleep(3.0)

        # 2. Walk the grid. Each Tab moves between the role/GPU toggles the footer
        # describes. The bursts are long deliberately: they are this reel's only
        # driver-paced motion, and a six-key nudge produced four measurable frames.
        s.key(*(["Tab"] * 16), after=0.05)
        time.sleep(1.6)
        s.key("space", after=0.8)
        time.sleep(1.4)
        s.key(*(["Tab"] * 14), after=0.05)
        time.sleep(1.2)
        s.key("space", after=0.8)
        time.sleep(1.6)

        # 3. Replicas: put a second embedder on the other card.
        s.key(*(["Tab"] * 12), after=0.05)
        time.sleep(0.8)
        s.key("+", after=0.9)
        time.sleep(1.6)

        # 4. Preview the plan, then apply it.
        s.key("C-r", after=1.2)
        time.sleep(3.0)
        s.key("C-s", after=1.2)
        time.sleep(3.5)

        # 5. Back to automatic, so the reel does not imply this is required.
        s.key("C-x", after=1.2)
        time.sleep(3.0)

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
                              else "/tmp/tui-manual-placement.cast")))
