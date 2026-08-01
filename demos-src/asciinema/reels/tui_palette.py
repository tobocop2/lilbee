#!/usr/bin/env python3
"""tui-palette: the command surface, and every screen it reaches.

Expanded past the shipped reel: it must cover ALL surfaces, including the two the old
ring never had (Fleet/placement and Sessions), and it must show a real ``/add`` landing
in the Task Center rather than only naming the command.

The palette is the through-line. It is a navigator, not a one-shot, so it drives the tour
wherever it can. Fleet and Sessions have no palette entry on this build -- searching
"fleet" and "placement" both return "No matches found" -- so those two use the direct
bindings the footer advertises, ^g and ^o.

Beat notes that cost takes to learn:
  * The palette must be confirmed open before typing a filter. If it is not, the filter
    text lands in the chat input, and a later "/help" appends to it and submits as a chat
    message instead of a command.
  * Escape followed by a key needs the driver's escape floor; Textual's ESCAPE_DELAY
    turns a fast pair into alt+<key>, which silently does nothing for non-priority keys.
  * The shipped reel cycles the theme and never restores, so it spends its last two
    thirds on a stale teal and closes there. This one restores rose-pine by name.
"""
from __future__ import annotations

import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "tui-palette"
COLS, ROWS = 128, 41
MUST_STRINGS = ("Slash Commands", "Search for commands", "Placement")

STAGE = pathlib.Path.home() / ".cache/lilbee-reel/lilbee"


def _palette(s: drive.Session, term: str, *, take: bool, dwell: float = 0.9) -> None:
    """Open the palette, filter, then either take the top hit or back out."""
    s.key("C-p", after=0.8)
    s.wait_for(r"Search for commands", timeout=15)
    s.type_text(term, rate=0.035)
    time.sleep(dwell)
    if take:
        s.key("Enter", after=1.0)
    else:
        s.esc()
        time.sleep(0.4)


def record(cast: pathlib.Path) -> dict:
    if not (STAGE / "data").exists():
        raise SystemExit("staged root missing; build it first (bb-unp94)")
    # A file the staged corpus does not already hold, so the add does real work on camera.
    incoming = STAGE / "incoming"
    incoming.mkdir(exist_ok=True)
    src = pathlib.Path.home() / "projects/lilbee/docs/usage.md"
    if src.exists():
        shutil.copy(src, incoming / "usage.md")

    s = drive.Session("reel-palette", COLS, ROWS, cast)
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    s.start("lilbee", env={"LILBEE_DATA": str(STAGE)})
    try:
        timings["boot"] = s.wait_for(r"personal encyclopedia", timeout=90)
        time.sleep(1.2)
        s.repaint()
        s.wait_for(r"personal encyclopedia", timeout=20)
        time.sleep(0.5)
        s.mark("boot_end")
        s.esc(2)
        time.sleep(0.5)

        # 1. The palette, unfiltered: every registered action at once.
        s.key("C-p", after=0.8)
        s.wait_for(r"Search for commands", timeout=15)
        time.sleep(1.1)
        s.esc()
        time.sleep(0.4)

        # 2-5. The palette as navigator, one surface at a time.
        for term, marker in (("catalog", r"Grid|List|Discover"),
                             ("status", r"Configuration|Indexed|Documents"),
                             ("settings", r"Ingest|Generation|Retrieval"),
                             ("task", r"Background Tasks")):
            _palette(s, term, take=True)
            s.wait_for(marker, timeout=25)
            time.sleep(1.1)

        # 6. Fleet / GPU placement. No palette entry exists, so use its binding.
        s.key("C-g", after=1.0)
        s.wait_for(r"Placement", timeout=25)
        time.sleep(1.4)
        # Wait for the overlay to actually go. A drawer that is still up swallows the
        # next beat's keys, and the failure looks like a missing INSERT much later.
        s.esc()
        s.wait_for(r"Placement", absent=True, timeout=15)
        time.sleep(0.4)

        # 7. Sessions.
        s.key("C-o", after=1.0)
        s.wait_for(r"Filter conversations|No saved conversations", timeout=25)
        time.sleep(1.2)
        # ^o is a TOGGLE. Escape and q both leave the drawer up (verified); only a second
        # ^o closes it, and the reel otherwise fails several beats later with a missing
        # INSERT because the drawer is still swallowing keys.
        s.key("C-o", after=1.0)
        s.wait_for(r"Filter conversations", absent=True, timeout=15)
        time.sleep(0.5)

        # 8. Back to Chat, then a real /add so the Task Center does visible work.
        s.goto("Chat", forward=False, limit=8, marker=r"personal encyclopedia")
        time.sleep(0.5)
        s.key("i", after=0.5)
        s.wait_for(r"INSERT", timeout=10)
        s.key("C-u", after=0.3)
        s.type_text(f"/add {incoming}", rate=0.035)
        time.sleep(0.6)
        s.key("Enter", after=1.0)
        s.esc()
        time.sleep(0.3)
        s.key("t", after=0.8)
        s.wait_for(r"Background Tasks", timeout=25)
        # Let the ingest visibly run and finish rather than cutting mid-bar.
        s.wait_for(r"caught up|done", timeout=180)
        time.sleep(1.2)

        # 9. The slash-command catalogue.
        s.goto("Chat", forward=False, limit=8, marker=r"personal encyclopedia")
        time.sleep(0.4)
        s.key("i", after=0.5)
        s.wait_for(r"INSERT", timeout=10)
        s.key("C-u", after=0.3)
        s.type_text("/help", rate=0.035)
        time.sleep(0.7)
        s.key("Enter", after=0.8)
        s.wait_for(r"Slash Commands", timeout=30)
        time.sleep(1.8)
        s.esc()
        time.sleep(0.6)

        # 10. Theme cycle, then the explicit restore the shipped reel never does.
        _palette(s, "theme", take=True, dwell=0.8)
        time.sleep(1.3)
        s.key("i", after=0.5)
        s.key("C-u", after=0.3)
        s.type_text("/theme rose-pine", rate=0.035)
        time.sleep(0.6)
        s.key("Enter", after=1.0)
        time.sleep(1.6)

        timings["total"] = time.monotonic() - t0
        s.mark("payload_end")
        time.sleep(1.0)
    finally:
        s.kill()
        shutil.rmtree(incoming, ignore_errors=True)
    timings["marks"] = dict(s.marks)
    timings["motion_spans"] = list(s.motion_spans)
    return timings


if __name__ == "__main__":
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tui-palette.cast")))
