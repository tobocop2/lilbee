#!/usr/bin/env python3
"""wiki-solar: lilbee turns a folder of documents into a browsable, cited wiki.

Ten Solar System articles, wikified into per-entity pages, browsed in the TUI:
an auto-generated encyclopedia where every claim carries a verified citation.
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "wiki-solar"
COLS, ROWS = 128, 41
DATA = pathlib.Path.home() / "lilbee-wiki-reel" / "data"
MUST_STRINGS = ("Jupiter", "Great Red Spot")

ENV = {
    "LILBEE_DATA": str(DATA),
    "LILBEE_WIKI": "true",
    "LILBEE_CHAT_MODEL": "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
    "LILBEE_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf",
}


def record(cast: pathlib.Path) -> dict:
    if not (DATA / "wiki").exists():
        raise SystemExit("build the solar wiki first (lilbee wiki build)")
    s = drive.Session("reel-wiki", COLS, ROWS, cast)
    timings: dict[str, float] = {}
    s.start("lilbee", env=ENV)
    try:
        timings["boot"] = s.wait_for(r"personal encyclopedia", timeout=120)
        time.sleep(1.5)
        s.mark("boot_end")
        s.esc(2)
        time.sleep(0.6)

        # Ring-navigate to the Wiki view. goto() overshoots because the wiki tree mounts
        # slowly (engine warmup), so step the ring explicitly and wait for the tree.
        for _ in range(5):
            s.key("]", after=0.8)
        s.wait_for(r"Filter pages|▼ Entities|Heliology", timeout=90)
        time.sleep(1.8)

        # Scroll the auto-generated page list (the sustained motion span), then open the
        # Jupiter page: the right pane renders the generated page with its citations.
        s.mark("browse_start")
        s.key("g", after=0.5)
        s.key(*(["j"] * 14), after=0.07)
        time.sleep(0.9)
        s.key("enter", after=1.0)
        timings["page"] = s.wait_for(r"fifth planet|Galilean|Great Red Spot|115 moons", timeout=25)
        time.sleep(3.6)
        s.mark("browse_end")
    finally:
        s.kill()
    return timings
