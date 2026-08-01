#!/usr/bin/env python3
"""wiki-lazy: browse a cited encyclopedia, then watch lilbee write a new page on demand.

The reel is all wiki. It opens straight on the Wiki view, pages through several
already-written, cited articles, then searches for a page that does not exist
yet, asks lilbee to write it, and follows the job in the Task Center while the
70B generates it -- ending on the finished, cited page. Recorded on a GPU pod;
the generation window is marked so post-trim compresses it instead of shipping a
progress bar.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "wiki-lazy"
COLS, ROWS = 128, 44
MUST_STRINGS = ("not written", "Write this page?", "Background Tasks", "gas giant")
FORBID_STRINGS = ("Traceback", "Could not write")
SPEED_WINDOWS = ("gen",)
SPEED_FACTOR = 16

LILBEE = os.environ.get("REEL_LILBEE_CMD", "lilbee")
DATA = os.environ.get("REEL_LILBEE_DATA", str(pathlib.Path.home() / "lilbee-wiki-reel" / "data"))

ENV = {
    "LILBEE_DATA": DATA,
    "LILBEE_WIKI": "true",
}
for _k in ("LILBEE_CHAT_MODEL", "LILBEE_EMBEDDING_MODEL", "LILBEE_CHAT_N_CTX_TARGET", "VIRTUAL_ENV"):
    if os.environ.get(_k):
        ENV[_k] = os.environ[_k]

# Absolute row of each page to browse in the unfiltered tree (Index, Log,
# Concepts+4, Entities, then entities alphabetically). Real, already-written
# planet articles. Verified against the live tree.
BROWSE_ROWS = (15, 29, 34)  # Earth, Mars, Neptune


def _to_wiki(s) -> None:
    """From the chat home, step to the Wiki view without lingering elsewhere."""
    for _ in range(5):
        s.key("]", after=0.6)
    s.wait_for(r"Not written yet|Filter pages", timeout=90)
    time.sleep(1.2)


def _open_row(s, row: int) -> None:
    """Open the page at absolute tree ``row`` from the top. The j burst is real
    driver motion, so it also feeds motion_fps."""
    s.key("g", after=0.4)
    s.key(*(["j"] * row), after=0.05)
    s.key("enter", after=0.9)


def _open_stub_by_search(s, needle: str) -> None:
    """Filter to one unwritten stub and open it (its group renders collapsed)."""
    s.key("/", after=0.4)
    s.key("C-u", after=0.2)
    s.key(*needle, after=0.11)
    time.sleep(1.0)
    s.key("Tab", after=0.4)
    s.key("g", after=0.35)
    s.key("j", after=0.35)
    s.key("j", after=0.35)
    s.key("l", after=0.5)
    s.key("j", after=0.4)
    s.key("enter", after=0.9)


def record(cast: pathlib.Path) -> dict:
    s = drive.Session("reel-wikilazy", COLS, ROWS, cast)
    t: dict[str, float] = {}
    s.start(LILBEE, env=ENV)
    try:
        s.wait_for(r"personal encyclopedia|Welcome to lilbee", timeout=150)
        time.sleep(1.2)
        if "Welcome to lilbee" in s.screen():
            s.esc(2)
            time.sleep(1.6)
        else:
            s.esc(2)
            time.sleep(0.6)
        s.mark("boot_end")

        _to_wiki(s)

        # Page through several already-written, cited articles.
        s.mark("browse_start")
        for row in BROWSE_ROWS:
            _open_row(s, row)
            s.wait_for(r"faithfulness|\[\^src", timeout=20)
            time.sleep(2.6)
        s.mark("browse_end")

        # Now a page that has not been written. Search finds it as a dim stub.
        _open_stub_by_search(s, "jupiter")
        s.wait_for(r"Write this page\?", timeout=25)
        time.sleep(2.6)

        # Confirm, then follow the job in the Task Center so the work is visible:
        # the running row's sweeping bar is the generation happening on camera.
        s.mark("gen_start")
        s.key("y", after=0.7)
        s.wait_for(r"Write Jupiter|Writing", timeout=20)
        time.sleep(0.6)
        s.key("t", after=0.8)
        s.wait_for(r"Background Tasks", timeout=15)
        time.sleep(2.0)
        t["gen"] = s.wait_for(r"1 done", timeout=400)
        time.sleep(1.6)
        s.mark("gen_end")

        # Step back to the Wiki (Tasks is next to Wiki in the ring; the Task
        # Center's own back key goes to Chat), then open the freshly written page.
        s.key("]", after=0.9)
        s.wait_for(r"Filter pages|Entities", timeout=20)
        time.sleep(0.6)
        _open_generated_page(s)
        s.wait_for(r"gas giant|fifth planet", timeout=25)
        time.sleep(4.0)
        s.mark("payload_end")
        time.sleep(0.8)
    finally:
        s.kill()
    t["marks"] = dict(s.marks)
    t["motion_spans"] = list(s.motion_spans)
    return t


def _open_generated_page(s) -> None:
    """Open Jupiter after generation, when it is a real page under Entities.

    Filter to it so the tree holds only Index, Log and an Entities group whose
    single child is the new page, then jump to the bottom (the page) and open it.
    """
    s.key("/", after=0.4)
    s.key("C-u", after=0.2)
    s.key(*"jupiter", after=0.11)
    time.sleep(1.0)
    s.key("Tab", after=0.4)
    s.key("G", after=0.6)
    s.key("enter", after=1.0)
