#!/usr/bin/env python3
"""tui-tour: the app stays yours while a model is generating.

The claim is asynchrony, so the reel has to prove it rather than assert it: the question
goes in, and then every other screen gets used while the answer is still streaming. If
the UI blocked during generation this reel could not be recorded at all.

Deliberately does not compress the generation window. Everywhere else the wait is the
boring part; here the wait is the subject, and speeding it up would hide the very thing
being demonstrated. What fills it is navigation, not a progress bar.

Storyboarded for a pod so the fleet panel would show more than one card. Recorded on the
laptop instead: the placement drawer states "One graphics card: everything runs here",
which is true of most people watching, and the async claim does not depend on card count.
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "tui-tour"
COLS, ROWS = 128, 41
MUST_STRINGS = ("oil capacity", "Discover", "Background Tasks", "Sources:")
TAIL_FORBID = ("Cancel stream",)
# No SPEED_WINDOWS: see above.
SPEED_WINDOWS = ()

STAGE = pathlib.Path.home() / ".cache/lilbee-reel/lilbee"
QUESTION = "what's my oil capacity?"


def record(cast: pathlib.Path) -> dict:
    if not (STAGE / "data").exists():
        raise SystemExit("staged root missing; build it first")

    s = drive.Session("reel-tour", COLS, ROWS, cast)
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    s.start("lilbee", env={"LILBEE_DATA": str(STAGE)})
    try:
        timings["boot"] = s.wait_for(r"personal encyclopedia", timeout=120)
        time.sleep(1.2)
        s.repaint()
        s.wait_for(r"personal encyclopedia", timeout=20)
        time.sleep(0.6)
        s.mark("boot_end")

        # 1. Ask, then leave insert mode while it is still generating.
        s.ask(QUESTION, rate=0.045)
        s.wait_for(r"Cancel stream", timeout=60)
        s.esc()
        time.sleep(0.6)

        # 2. Use the app. Every one of these happens while the answer is streaming.
        s.key("]", after=0.6)
        s.wait_for(r"Discover", timeout=40)
        time.sleep(1.2)
        for tab in ("2", "3", "6"):
            s.key(tab, after=0.5)
            time.sleep(1.0)
        s.key(*(["j"] * 18), after=0.045)
        time.sleep(0.8)

        s.key("]", after=0.6)          # Status
        time.sleep(1.6)
        s.key("]", after=0.6)          # Settings
        time.sleep(1.2)
        s.key(*(["j"] * 20), after=0.045)
        time.sleep(0.8)
        s.key(">", after=0.6)
        time.sleep(1.0)
        s.key(*(["j"] * 16), after=0.045)
        time.sleep(0.8)

        s.key("]", after=0.6)          # Tasks
        s.wait_for(r"Background Tasks", timeout=25)
        time.sleep(1.4)

        # 3. Back to Chat for the finished answer.
        s.goto("Chat", forward=False, limit=8, marker=r"Slash commands")
        time.sleep(0.8)
        timings["answer"] = s.await_answer()
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
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tui-tour.cast")))
