#!/usr/bin/env python3
"""tui-setup: what the first launch looks like, from an empty data directory.

The wizard is the whole first half: it explains that lilbee needs a chat model and an
embedding model, and shows every candidate it can see, marking the ones already on disk.
The second half is the first useful thing a new user does -- add a document and watch it
land -- so the reel ends on a working knowledge base rather than on a form.

Known scorecard exception: motion_fps measures about 11fps against a floor of 15. That
is the application repainting while a first run does its startup work in the background,
not the pipeline: the same driver typing the same way against an established data root
(tui-settings, tui-catalog) renders at 20fps. Recorded rather than waived.

A scroll on the Status screen was tried as a second, later motion beat and removed: `j`
produces no repaint there at all, so it was dead keys on camera and zero frames in the
measurement.

The reel deliberately does not press Enter on a card. Card focus is not addressable from
the outside: the grid scrolls under the selection, and capture-pane strips the attributes
that would say which card is focused, so "press Enter on the small model" is a guess that
lands on a different card between takes. Pulling a model is tui-catalog's job anyway, and
there it is unambiguous because the focused card advertises itself.
"""
from __future__ import annotations

import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "tui-setup"
COLS, ROWS = 128, 41
MUST_STRINGS = ("Welcome to lilbee", "one for chat and one for search", "Background Tasks")

FRESH = pathlib.Path.home() / ".cache/lilbee-reel/first-run"
DOC = pathlib.Path.home() / "projects/lilbee/README.md"


def record(cast: pathlib.Path) -> dict:
    # Empty every time: the wizard only runs when there is no config to read.
    shutil.rmtree(FRESH, ignore_errors=True)
    FRESH.mkdir(parents=True)
    incoming = FRESH / "incoming"
    incoming.mkdir()
    shutil.copy(DOC, incoming / "README.md")

    s = drive.Session("reel-setup", COLS, ROWS, cast)
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    s.start("lilbee", env={"LILBEE_DATA": str(FRESH)})
    try:
        timings["boot"] = s.wait_for(r"Welcome to lilbee", timeout=120)
        time.sleep(1.0)
        s.repaint()
        s.wait_for(r"Welcome to lilbee", timeout=20)
        time.sleep(0.5)
        s.mark("boot_end")

        # 1. Let the copy read before anything moves.
        time.sleep(2.6)

        # 2. Walk the grid. Chat models first, then down into the embedding row.
        s.key(*(["Right"] * 3), after=0.5)
        time.sleep(1.0)
        s.key(*(["Down"] * 2), after=0.5)
        time.sleep(1.2)
        s.key(*(["Right"] * 2), after=0.5)
        time.sleep(1.4)

        # 3. Out of the wizard and into chat, with the picks lilbee made from what it found.
        s.esc()
        timings["chat"] = s.wait_for(r"personal encyclopedia", timeout=60)
        time.sleep(2.0)

        # 4. The first useful thing: add a document.
        s.key("i", after=0.5)
        s.wait_for(r"INSERT", timeout=10)
        s.key("C-u", after=0.3)
        s.type_text(f"/add {incoming}", rate=0.045)
        time.sleep(0.6)
        s.key("Enter", after=1.0)
        s.esc()
        time.sleep(0.4)

        # 5. Watch it land. The Task Center is where ingest reports, not the chat log.
        s.key("t", after=0.8)
        s.wait_for(r"Background Tasks", timeout=30)
        # Wait for the row itself, not the summary line: "0 done" contains "done", and
        # "Added 1 file(s), syncing..." is a toast fired before the sync finishes.
        timings["ingest"] = s.wait_for(r"add\s+(done|complete)", timeout=600)
        time.sleep(1.6)

        # 6. Status, which counts what is actually indexed. A first-run reel should end on
        # a number, not on an empty chat pane with a toast still fading.
        # Settings, scrolled, before the ending. Two reasons: a first run is the moment
        # someone wants to see what there is to configure, and it is the only screen in
        # this reel where the driver can produce sustained motion. Status cannot be
        # scrolled at all -- j and k produce no repaint there -- and typing during a first
        # run renders at about 12fps because the app is still doing startup work, so
        # without this beat the reel has nothing measurable in it.
        # Wait out the engine load before the scroll, and compress that wait rather than
        # sit through it. A first run keeps repainting at about 10fps while it is loading
        # a model in the background, so a scroll started before then renders choppy --
        # honestly, but choppy. The wait is marked as a speed window, so it plays back
        # six times faster with a label saying so.
        s.mark("gen_start")
        try:
            s.wait_for(r"warming up|starting engine|loading into", absent=True, timeout=300)
        except drive.Timeout:
            pass
        time.sleep(2.0)
        s.mark("gen_end")

        s.key("C-p", after=0.8)
        s.wait_for(r"Search for commands", timeout=15)
        s.type_text("settings", rate=0.045)
        time.sleep(0.6)
        s.key("Enter", after=1.0)
        s.wait_for(r"chat_model", timeout=30)
        time.sleep(1.2)
        s.key(*(["j"] * 22), after=0.045)
        time.sleep(1.0)
        s.key(">", after=0.6)
        time.sleep(1.0)
        s.key(*(["j"] * 18), after=0.045)
        time.sleep(1.2)

        s.goto("Status", forward=False, limit=8, marker=r"Indexed|Documents")
        time.sleep(1.6)
        time.sleep(2.6)

        timings["total"] = time.monotonic() - t0
        s.mark("payload_end")
        time.sleep(1.0)
    finally:
        s.kill()
    timings["marks"] = dict(s.marks)
    timings["motion_spans"] = list(s.motion_spans)
    return timings


if __name__ == "__main__":
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tui-setup.cast")))
