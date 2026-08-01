#!/usr/bin/env python3
"""tui-crawl-site: a whole section of a site at depth 1, then a question that spans it.

Same modal as tui-crawl with the opposite settings: recursion left on, a page cap so the
crawl is bounded and the reel ends, depth 1 so the answer has to come from more than the
seed page. The Task Center shows several pages landing rather than one.

Deliberately a different article from tui-crawl. Both reels pointed at the Caprice page
for one take, which made the site reel look like the single-page reel with a bigger
number in the page cap; the whole distinction it exists to draw was invisible. Full-size
car is a hub article whose depth-1 neighbourhood is a set of related models, so the
answer has to come from more than the page that was typed in.
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402
from reels.tui_crawl import fresh_root  # noqa: E402

# Qwen3 8B here: this reel synthesises an answer across several crawled pages,
# which is where the larger model earns its load time.
CHAT_MODEL = "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"

NAME = "tui-crawl-site"
COLS, ROWS = 128, 41
MUST_STRINGS = ("Recursive", "Max pages", "Full-size_car")
# Nothing may still be generating when the reel stops.
TAIL_FORBID = ("Cancel stream",)


ROOT = pathlib.Path.home() / ".cache/lilbee-reel/crawl-site"
URL = "https://en.wikipedia.org/wiki/Full-size_car"
MAX_PAGES = "5"
QUESTION = "what defines a full-size car, and name two examples?"


def record(cast: pathlib.Path) -> dict:
    root = fresh_root(ROOT, CHAT_MODEL)

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
        # Placement stays open through the answer: it shows which card the model is on
        # while it is generating, which is the part of the story a chat pane alone
        # cannot tell.
        s.key("C-g", after=1.0)
        s.wait_for(r"Placement", timeout=25)
        time.sleep(1.4)
        s.ask(QUESTION, rate=0.04)
        s.mark("gen_start")
        timings["answer"] = s.await_answer()
        s.mark("gen_end")
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
