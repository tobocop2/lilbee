#!/usr/bin/env python3
"""tui-crawl-site: a whole section of a site at depth 1, then a question that spans it.

Same modal as tui-crawl with the opposite settings: recursion left on, a page cap so the
crawl is bounded and the reel ends, depth 1 so the answer has to come from more than the
seed page. The Task Center shows several pages landing rather than one.
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402
from reels.tui_crawl import fresh_root  # noqa: E402

NAME = "tui-crawl-site"
COLS, ROWS = 128, 41
MUST_STRINGS = ("Recursive", "Max pages", "Caprice")

ROOT = pathlib.Path.home() / ".cache/lilbee-reel/crawl-site"
URL = "https://en.wikipedia.org/wiki/Chevrolet_Caprice"
MAX_PAGES = "5"
QUESTION = "which engines were offered in the Caprice, and what replaced it?"


def record(cast: pathlib.Path) -> dict:
    root = fresh_root(ROOT)

    s = drive.Session("reel-crawlsite", COLS, ROWS, cast)
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    s.start("lilbee", env={"LILBEE_DATA": str(root)})
    try:
        timings["boot"] = s.wait_for(r"personal encyclopedia", timeout=120)
        time.sleep(1.2)
        s.repaint()
        s.wait_for(r"personal encyclopedia", timeout=20)
        time.sleep(0.6)
        s.mark("boot_end")

        s.key("i", after=0.5)
        s.wait_for(r"INSERT", timeout=10)
        s.key("C-u", after=0.3)
        s.type_text("/crawl", rate=0.05)
        time.sleep(0.6)
        s.key("Enter", after=1.0)
        s.wait_for(r"Recursive", timeout=20)
        time.sleep(1.0)

        s.type_text(URL, rate=0.035)
        time.sleep(1.0)

        # Recursive stays on. Only the browser toggle comes off: a Chromium render buys
        # nothing on Wikipedia and costs memory the reel would then have to explain.
        s.key("Tab", after=0.6)
        s.key("Tab", after=0.6)
        s.key("Space", after=0.8)
        s.key("Tab", after=0.6)
        s.key("C-u", after=0.3)
        s.type_text(MAX_PAGES, rate=0.06)
        time.sleep(0.5)
        s.key("Tab", after=0.6)
        s.key("C-u", after=0.3)
        s.type_text("1", rate=0.06)
        time.sleep(0.8)

        s.key("Tab", after=0.6)
        s.key("Enter", after=1.0)
        s.wait_for(r"Recursive", absent=True, timeout=20)
        time.sleep(0.8)

        s.esc()
        time.sleep(0.3)
        s.key("t", after=0.8)
        s.wait_for(r"Background Tasks", timeout=30)
        timings["crawl"] = s.wait_for(r"(crawl|add)\s+(done|complete)|all caught up",
                                      timeout=1200)
        time.sleep(2.2)

        s.goto("Chat", forward=False, limit=8, marker=r"personal encyclopedia")
        time.sleep(0.6)
        s.key("i", after=0.5)
        s.wait_for(r"INSERT", timeout=10)
        s.key("C-u", after=0.3)
        s.type_text(QUESTION, rate=0.04)
        time.sleep(0.8)
        s.key("Enter", after=0.8)
        timings["answer"] = s.wait_for(r"Sources:", timeout=900)
        time.sleep(4.0)

        timings["total"] = time.monotonic() - t0
        s.mark("payload_end")
        time.sleep(1.0)
    finally:
        s.kill()
    timings["marks"] = dict(s.marks)
    timings["motion_spans"] = list(s.motion_spans)
    return timings


if __name__ == "__main__":
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tui-crawl-site.cast")))
