#!/usr/bin/env python3
"""wiki-solar: lilbee turns a folder of documents into a browsable, cited wiki.

Ten Solar System articles, wikified by Llama-3.3-70B into per-entity pages,
browsed in the TUI: an auto-generated encyclopedia where every claim carries a
verified citation and cross-links to related pages via [[wiki links]].
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "wiki-solar"
COLS, ROWS = 128, 44
DATA = pathlib.Path.home() / "lilbee-wiki-reel" / "data"
MUST_STRINGS = ("Jupiter", "gas giant")

ENV = {
    "LILBEE_DATA": str(DATA),
    "LILBEE_WIKI": "true",
    "LILBEE_CHAT_MODEL": "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
    "LILBEE_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf",
}


def record(cast: pathlib.Path) -> dict:
    if not (DATA / "wiki").exists():
        raise SystemExit("build the solar wiki first")
    s = drive.Session("reel-wiki", COLS, ROWS, cast)
    timings: dict[str, float] = {}
    s.start("lilbee", env=ENV)
    try:
        s.wait_for(r"personal encyclopedia|Welcome to lilbee", timeout=120)
        time.sleep(1.2)
        if "Welcome to lilbee" in s.screen():
            s.esc(2)
            time.sleep(1.6)
        else:
            s.esc(2)
            time.sleep(0.7)
        s.mark("boot_end")

        # Ring-navigate to the Wiki view (goto overshoots on the slow-mounting tree, so
        # step the ring explicitly and wait for the page tree to mount).
        for _ in range(5):
            s.key("]", after=0.8)
        s.wait_for(r"Filter pages|▼ Entities", timeout=90)
        time.sleep(1.8)

        # Scroll the auto-generated page list (the sustained motion span reveals the
        # breadth: planets, moons, spacecraft, astronomers), then open the Jupiter page.
        s.mark("browse_start")
        s.key("g", after=0.6)
        s.key(*(["j"] * 27), after=0.06)
        time.sleep(1.0)
        s.key("enter", after=1.0)
        timings["page"] = s.wait_for(r"gas giant|fifth planet|Great Red|Galilean", timeout=25)
        time.sleep(4.0)
        s.mark("browse_end")
    finally:
        s.kill()
    return timings
