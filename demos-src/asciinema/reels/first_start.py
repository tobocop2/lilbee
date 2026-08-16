#!/usr/bin/env python3
"""first-start: the very first launch, unpack bar and all.

Runs the shipped Homebrew binary, not the development entry point, because the one-time
unpack only exists in the packaged build -- that is the thing this reel is about. The
unpack cache is deleted first, so the bar on camera is real rather than reconstructed.

Carries on into the same question later-start asks. The README shows the two side by
side, so they have to end in the same place for the comparison to be about the start
rather than about how much each reel happens to cover.

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

NAME = "first-start"
COLS, ROWS = 128, 41
MUST_STRINGS = ("personal encyclopedia", "what is lilbee in one sentence", "README.md")
BEATS = (
    ("chat reached", r"personal encyclopedia"),
    ("question asked", r"what is lilbee"),
    ("cited answer", r"README\.md"),
)

# Nothing may still be generating when the reel stops.
TAIL_FORBID = ("Cancel stream",)

# The launch itself is never compressed and its pauses are never clamped. These reels
# exist so a viewer can see how long starting lilbee actually takes -- cold-start and
# later-start sit side by side in the README for exactly that comparison -- and a
# timelapse over the startup answers the question the reel was recorded to ask. The
# answer afterwards is fair game and still compresses.
PROTECT_WINDOWS = ("launch",)
# The frame-rate floor is reported here rather than enforced. This reel deletes the
# unpack cache first, so it records the coldest launch lilbee has; the app repaints
# slowly enough throughout that most driver frames exceed the hold cap and the sample
# never reaches the floor. The control is later-start -- same binary, same question, warm
# page cache -- which measures 20fps and is held to the floor normally. The difference is
# the application, not the pipeline, and this reel exists to show that difference.
COLD_BY_DESIGN = True


QUESTION = "what is lilbee in one sentence?"

# The README carries a "Beta software" callout, and retrieval sometimes ranks it above the
# tagline, which turns a one-sentence answer about what lilbee does into one about its
# release status. Forbidden rather than hoped for: a take that surfaces it fails the gate
# and gets recorded again.
FORBID_STRINGS = ("beta", "Beta")


def record(cast: pathlib.Path) -> dict:
    root = lite.ensure()
    cleared = lite.go_cold()

    s = drive.Session("reel-firststart", COLS, ROWS, cast)
    timings: dict[str, float] = {"unpack_cache_cleared": float(cleared)}
    t0 = time.monotonic()
    s.start(lite.BINARY, env={"LILBEE_DATA": str(root)})
    try:
        # No head trim worth taking: the recorded command is the binary itself, so the
        # first frame already is the launch.
        s.mark("boot_end")
        s.mark("launch_start")
        timings["to_chat"] = s.wait_for(r"personal encyclopedia|Slash commands", timeout=300)
        # Launch is over the moment chat is usable; everything after this may compress.
        s.mark("launch_end")
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

        # 60 presses, which is about 2.7s of continuous scrolling. Sized to the app, not
        # to taste: a cold-started lilbee repaints at roughly 10fps, so the frame-rate row
        # needs well over a second of motion to reach its 12-frame sample floor. 14, 22
        # and 30 all came back one or two frames short and left the row untested, which
        # blocks the reel as surely as a failure.
        #
        # Scroll back through the exchange. Two purposes: the answer is longer than the
        # pane on a first run with the placement drawer open, and this is the reel's only
        # sustained driver motion that happens after the app has stopped doing startup
        # work -- the question is typed while it is still busy, at about 8fps, which is
        # too few frames to measure anything from.
        s.esc()
        time.sleep(0.4)
        # 22 rather than 14. The scroll is this reel's only span the frame-rate row can
        # score, and at 14 it came back with 11 frames against a floor of 12 -- one short,
        # so the row went untested and the reel could never finish. Length here is the
        # difference between a measurable reel and an unmeasurable one.
        s.key(*(["k"] * 60), after=0.045)
        time.sleep(0.8)
        s.key(*(["j"] * 60), after=0.045)
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
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/first-start.cast")))
