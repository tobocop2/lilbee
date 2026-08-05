#!/usr/bin/env python3
"""tui-settings: every setting lilbee exposes, without leaving the terminal.

The shipped reel walks eight panes. This build has ten -- Local-Servers and Memory landed
since -- so the walk is driven off ``>`` until the last pane rather than a fixed count.

Two constraints carried over from the retired tape, both learned the hard way:
  * Never Tab into a field. Focusing an Input makes ``>``, ``q`` and ``/`` type as literal
    characters into it, and the rest of the reel is garbage typed into a text box.
  * Stay on screen-level bindings: ``>`` cycles panes, ``j``/``k`` scroll without focusing.

Scrolling is sent as a fast burst rather than one keypress every 800ms. A burst reads as
a scroll instead of a slideshow, and it is also the only motion in a reel that otherwise
holds still, so it is what makes the frame rate measurable at all.

Each pane scrolls down and stays there. Every pane keeps its own scroll offset, so the
next one still opens at its top; scrolling back up doubled the frame count, and a
full-screen scroll compresses badly enough that those frames were most of the file size.
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "tui-settings"
COLS, ROWS = 128, 41
# Fields from panes deep in the walk. The tab strip names every pane on every frame, so
# asserting pane names would pass without the walk ever happening.
MUST_STRINGS = ("ollama_base_url", "crawl_render_mode", "num_ctx_max",
                "memory_token_budget")
BEATS = (
    ("models pane", r"chat_model"),
    ("local servers pane", r"ollama_base_url"),
    ("generation pane", r"num_ctx_max"),
    ("memory pane", r"memory_token_budget"),
)

STAGE = pathlib.Path.home() / ".cache/lilbee-reel/lilbee"

# (settle before scrolling, scroll presses) per pane, in `>` order.
PANES = [
    ("Models", 1.6, 8),
    ("Ingest", 1.3, 16),
    ("Local-Servers", 1.3, 4),
    ("Crawling", 1.3, 16),
    ("Generation", 1.3, 20),
    ("Retrieval", 1.3, 20),
    ("Display", 1.3, 6),
    ("Memory", 1.3, 8),
    ("API-Keys", 1.3, 6),
    ("System", 1.4, 12),
]

# 45ms between scroll steps. This is the reel's frame rate while it moves, not a taste
# call: agg emits one frame per content change, so a 90ms scroll renders at 10fps.
SCROLL_RATE = 0.045


def _scroll(s: drive.Session, presses: int, key: str = "j") -> None:
    s.key(*([key] * presses), after=SCROLL_RATE)


def record(cast: pathlib.Path) -> dict:
    if not (STAGE / "data").exists():
        raise SystemExit("staged root missing; build it first")

    s = drive.Session("reel-settings", COLS, ROWS, cast)
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    s.start("lilbee", env={"LILBEE_DATA": str(STAGE)})
    try:
        timings["boot"] = s.await_chat(timeout=90)
        time.sleep(1.2)
        s.repaint()
        s.wait_for(r"personal encyclopedia|Slash commands", timeout=20)
        time.sleep(0.5)
        s.mark("boot_end")
        s.esc(2)
        time.sleep(0.5)

        # Jump via the palette rather than walking the ring. Walking forward passes
        # through Catalog, which fetches the HF Hub index on entry and blocks the UI for
        # ~5s; the queued keys then all land at once and the reel opens on five seconds
        # of a frozen catalog. Filed separately -- the reel routes around it.
        s.key("C-p", after=0.8)
        s.wait_for(r"Search for commands", timeout=15)
        s.type_text("settings", rate=0.045)
        time.sleep(0.7)
        s.key("Enter", after=1.0)
        s.wait_for(r"chat_model", timeout=25)
        time.sleep(0.8)

        for i, (pane, settle, presses) in enumerate(PANES):
            if i:
                s.key(">", after=0.6)
                s.wait_for(pane, timeout=15)
            time.sleep(settle)
            _scroll(s, presses)
            time.sleep(0.7)

        timings["total"] = time.monotonic() - t0
        s.mark("payload_end")
        time.sleep(1.0)
    finally:
        s.kill()
    timings["marks"] = dict(s.marks)
    timings["motion_spans"] = list(s.motion_spans)
    return timings


if __name__ == "__main__":
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tui-settings.cast")))
