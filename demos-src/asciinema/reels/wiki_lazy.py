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
# The story, in order: read an already-written cited page, get prompted to write
# a missing one, watch it run in the Task Center, then read the freshly written
# page. A reel can pass every other gate and still show the wrong thing.
BEATS = (
    ("existing cited page read", r"Earth is the third"),
    ("prompted to write the missing page", r"Write this page\?"),
    ("generation running in the task center", r"Background Tasks"),
    ("freshly written page read", r"gas giant"),
)
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

# Absolute row of the first planet in the unfiltered tree (Index, Log, Concepts+4,
# Entities, then entities alphabetically) -- Earth. The rest are reached by filter.
EARTH_ROW = 15
# Planets to reach with the search filter after the first is paged to by hand.
# Both match exactly one written page (no adjective stub like "Saturnian"
# /"Venusian" to make the filter ambiguous), so jump-to-bottom lands on the page.
FILTER_PLANETS = ("neptune", "mercury")


def _to_wiki(s) -> None:
    """From the chat home, step to the Wiki view without lingering elsewhere."""
    for _ in range(5):
        s.key("]", after=0.6)
    s.wait_for(r"Not written yet|Filter pages", timeout=90)
    time.sleep(1.2)


def _open_row(s, row: int) -> None:
    """Open the page at absolute tree ``row`` from the top. The j burst is real
    driver motion; at this interval it renders near the 25fps cap so the scroll
    reads smoothly rather than choppy."""
    s.key("g", after=0.4)
    s.key(*(["j"] * row), after=0.038)
    s.key("enter", after=0.9)


def _open_page_by_filter(s, needle: str) -> None:
    """Filter to one written page and open it.

    A filter that matches a single real page leaves the Entities group expanded
    with that page as the last visible node, so jump-to-bottom lands on it.
    """
    s.key("/", after=0.4)
    s.key("C-u", after=0.2)
    s.key(*needle, after=0.11)
    time.sleep(1.0)
    s.key("Tab", after=0.4)
    s.key("G", after=0.5)
    s.key("enter", after=0.9)


def _scroll_through(s, chunks: int = 3, per: int = 14) -> None:
    """Read the whole article: focus the content pane, then scroll a screenful at
    a time, pausing on each so a section is read before the next.

    A fast burst per chunk moves the pane (a slow drip stalls it); the pauses are
    the reading. Only reliable once a generation has reloaded the wiki screen, so
    this is used on the freshly written page, not on a page opened at rest.
    """
    s.key("Tab", after=1.0)
    for _ in range(chunks):
        s.key(*(["Down"] * per), after=0.045)
        time.sleep(2.2)


def _open_stub_by_search(s, needle: str) -> None:
    """Filter to one unwritten stub and open it (its group renders collapsed).

    The query is typed just above the motion-span threshold so the discrete
    typing does not register as motion and drag motion_fps down; the smooth tree
    scroll is what that gate measures.
    """
    s.key("/", after=0.4)
    s.key("C-u", after=0.2)
    s.key(*needle, after=0.13)
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

        # 1. Page through the encyclopedia tree to an already-written article. The
        #    tree scroll is smooth (it repaints on every keystroke, unlike the
        #    async content pane), so it carries the reel's motion; the article is
        #    then read at rest (the content pane only scrolls after a generation
        #    reload).
        s.mark("browse_start")
        s.key("g", after=0.4)
        s.key(*(["j"] * 26), after=0.04)  # scroll down through the pages
        time.sleep(0.9)
        s.key("g", after=0.4)
        s.key(*(["j"] * EARTH_ROW), after=0.04)  # back up to Earth
        s.key("enter", after=0.9)
        s.wait_for(r"faithfulness|\[\^src", timeout=20)
        time.sleep(4.2)
        s.mark("browse_end")

        # 2. A page that has not been written: search finds it as a dim stub.
        _open_stub_by_search(s, "jupiter")
        s.wait_for(r"Write this page\?", timeout=25)
        time.sleep(2.4)

        # Confirm, then follow the job in the Task Center so the work is visible.
        s.mark("gen_start")
        s.key("y", after=0.7)
        s.wait_for(r"Write Jupiter|Writing", timeout=20)
        time.sleep(0.6)
        s.key("t", after=0.8)
        s.wait_for(r"Background Tasks", timeout=15)
        time.sleep(1.0)
        # Open the GPU panel so the card the model runs on is visible, its live
        # utilisation bars moving, while the page is written.
        s.key("C-g", after=0.8)
        time.sleep(1.6)
        t["gen"] = s.wait_for(r"1 done", timeout=400)
        time.sleep(1.6)
        s.key("C-g", after=0.6)  # close the drawer before leaving
        s.mark("gen_end")

        # 3. The freshly written page: step back to the Wiki (Tasks is next to it
        #    in the ring), open it, and read the whole generated article.
        s.key("]", after=0.9)
        s.wait_for(r"Filter pages|Entities", timeout=20)
        time.sleep(0.6)
        _open_generated_page(s)
        s.wait_for(r"gas giant|fifth planet", timeout=25)
        time.sleep(2.4)
        _scroll_through(s)
        time.sleep(2.4)
        s.mark("payload_end")
        time.sleep(1.0)
    finally:
        s.kill()
    t["marks"] = dict(s.marks)
    t["motion_spans"] = list(s.motion_spans)
    return t


def _scroll_page(s) -> None:
    """Scroll the content pane line by line so the whole article is shown.

    Focus method + scroll keys are set from the probe; the burst also feeds
    motion_fps.
    """
    s.key("Tab", after=0.5)
    s.key(*(["Down"] * 40), after=0.045)


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
