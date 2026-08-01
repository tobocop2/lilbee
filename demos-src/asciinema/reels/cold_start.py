#!/usr/bin/env python3
"""cold-start: the whole cold path, launch to cited answer, in one take.

The homepage version of the startup story. first-start stops at the chat screen and
later-start starts from a warm cache; this one carries the cold launch all the way
through the first question, so nothing about the slowest path is left off camera --
unpack, engine load, retrieval and generation are all in the same clock.

Known scorecard exception: motion_fps measures 10fps here against a floor of 15. That is
the application's repaint rate during a cold launch, not the pipeline's. The control is
later-start: same packaged binary, same question, same typing rate, warm page cache, and
it renders the identical beat at 20fps. Nothing in the driver or the renderer changes
between the two takes, so the difference is the cold process competing with its own model
load for disk. Recorded rather than waived -- the row stays red and this reel needs a
human call.
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402
import lite  # noqa: E402

NAME = "cold-start"
COLS, ROWS = 128, 41
MUST_STRINGS = ("personal encyclopedia", "what is lilbee in one sentence", "README.md")
# Nothing may still be generating when the reel stops.
TAIL_FORBID = ("Cancel stream",)


QUESTION = "what is lilbee in one sentence?"

# The README carries a "Beta software" callout, and retrieval sometimes ranks it above the
# tagline, which turns a one-sentence answer about what lilbee does into one about its
# release status. Forbidden rather than hoped for: a take that surfaces it fails the gate
# and gets recorded again.
FORBID_STRINGS = ("beta", "Beta")


def record(cast: pathlib.Path) -> dict:
    root = lite.ensure()
    timings: dict[str, float] = {"unpack_cache_cleared": float(lite.go_cold())}

    s = drive.Session("reel-coldstart", COLS, ROWS, cast)
    t0 = time.monotonic()
    s.start(lite.BINARY, env={"LILBEE_DATA": str(root)})
    try:
        s.mark("boot_end")
        timings["to_chat"] = s.wait_for(r"personal encyclopedia", timeout=300)
        # Let the process settle before typing. Straight after a cold unpack the app
        # is still warming and repaints at about 10fps, so a question typed the
        # instant the screen appears renders choppy -- honestly, but choppy. A
        # person would not type into a window that has just appeared either.
        time.sleep(6.0)

        # Placement stays open through the answer: it shows which card the model is on
        # while it is generating, which is the part of the story a chat pane alone
        # cannot tell.
        s.key("C-g", after=1.0)
        s.wait_for(r"Placement", timeout=25)
        time.sleep(1.4)
        s.ask(QUESTION, rate=0.045)
        s.mark("gen_start")
        timings["answer"] = s.await_answer()
        s.mark("gen_end")
        time.sleep(2.0)

        # Scroll back through the exchange. This is the reel's only sustained driver
        # motion that happens after the app has stopped doing startup work; the question
        # is typed while the engine is still loading, which renders too slowly and with
        # too few frames to measure anything from.
        s.esc()
        time.sleep(0.4)
        s.key(*(["k"] * 14), after=0.045)
        time.sleep(0.8)
        s.key(*(["j"] * 14), after=0.045)
        time.sleep(1.6)

        timings["total"] = time.monotonic() - t0
        s.mark("payload_end")
        time.sleep(1.0)
    finally:
        s.kill()
    timings["marks"] = dict(s.marks)
    timings["motion_spans"] = list(s.motion_spans)
    return timings


if __name__ == "__main__":
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/cold-start.cast")))
