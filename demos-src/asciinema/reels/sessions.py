#!/usr/bin/env python3
"""sessions: conversations persist, and you can go back to one.

Builds its own evidence. The drawer opens empty on a fresh root -- "No saved conversations
yet" -- so the reel holds two separate conversations on camera first, then opens the
drawer to find both and resumes the earlier one. Showing a pre-populated list would prove
nothing about whether anything was actually saved.

The drawer's own footer advertises `esc close` on this build, so Escape is used rather
than a second ^o. That is not true of the Fleet drawer, which only a second ^g will close.
"""
from __future__ import annotations

import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "sessions"
COLS, ROWS = 128, 41
MUST_STRINGS = ("oil capacity", "resume", "Sources:")
TAIL_FORBID = ("Cancel stream",)
SPEED_WINDOWS = ("gen", "gen2")

ROOT = pathlib.Path.home() / ".cache/lilbee-reel/sessions"
STAGE = pathlib.Path.home() / ".cache/lilbee-reel/lilbee"
FIRST = "what's my oil capacity?"
SECOND = "what should I do if the engine overheats?"


def record(cast: pathlib.Path) -> dict:
    # A copy of the staged root, so the drawer genuinely starts empty while the corpus is
    # already indexed. Recording against the staged root itself would show whatever
    # conversations previous takes happened to leave behind.
    shutil.rmtree(ROOT, ignore_errors=True)
    shutil.copytree(STAGE, ROOT)
    for stale in (ROOT / "sessions", ROOT / "data/sessions"):
        shutil.rmtree(stale, ignore_errors=True)

    s = drive.Session("reel-sessions", COLS, ROWS, cast)
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    s.start("lilbee", env={"LILBEE_DATA": str(ROOT)})
    try:
        timings["boot"] = s.wait_for(r"personal encyclopedia", timeout=120)
        time.sleep(1.2)
        s.repaint()
        s.wait_for(r"personal encyclopedia", timeout=20)
        time.sleep(0.6)
        s.mark("boot_end")

        # 1. First conversation.
        s.ask(FIRST, rate=0.05)
        s.mark("gen_start")
        timings["answer1"] = s.await_answer()
        s.mark("gen_end")
        time.sleep(2.0)

        # 2. Start a second one and ask something unrelated to the first.
        s.esc()
        time.sleep(0.4)
        s.key("C-n", after=1.2)
        time.sleep(1.0)
        s.ask(SECOND, rate=0.05)
        s.mark("gen2_start")
        timings["answer2"] = s.await_answer()
        s.mark("gen2_end")
        time.sleep(2.0)

        # 3. The drawer now has both. Resume the first.
        s.esc()
        time.sleep(0.4)
        s.key("C-o", after=1.2)
        s.wait_for(r"resume", timeout=25)
        time.sleep(2.0)
        s.key("Down", after=0.8)
        time.sleep(1.0)
        s.key("Enter", after=1.2)
        s.wait_for(r"oil capacity", timeout=30)
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
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sessions.cast")))
