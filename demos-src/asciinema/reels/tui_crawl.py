#!/usr/bin/env python3
"""tui-crawl: one web page into the knowledge base, then a question about it.

The `/crawl` modal has flattened since the retired tape was written: recursion, the
browser toggle, the page cap and the depth cap are all on one form now, with no Advanced
section to expand. Tab order is URL, Recursive, Use browser, Max pages, Depth cap, then
the Crawl button.

Both checkboxes ship enabled. This reel turns them off: recursion because the point is a
single page, and the browser because a Chromium render is neither needed for a static
article nor honest about what the default path costs.
"""
from __future__ import annotations

import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "tui-crawl"
COLS, ROWS = 128, 41
MUST_STRINGS = ("Recursive", "en.wikipedia.org", "9C1")
BEATS = (
    ("crawl modal", r"Recursive"),
    ("url entered", r"en\.wikipedia\.org"),
    ("crawl finished", r"(crawl|add)\s+(done|complete)|all caught up"),
    ("cited answer", r"Sources:"),
)

# Nothing may still be generating when the reel stops.
TAIL_FORBID = ("Cancel stream",)


ROOT = pathlib.Path.home() / ".cache/lilbee-reel/crawl"
URL = "https://en.wikipedia.org/wiki/Chevrolet_Caprice"
QUESTION = "When was the 9C1 Caprice police package introduced?"

# Gemma 4 E4B: a current multimodal pick from lilbee's own featured list, and a
# different voice from the Qwen answers elsewhere in the set. Q4_0 rather than Q4_K_M
# because that repo does not publish a Q4_K_M, which is a bug in the featured entry.
CHAT_MODEL = "ggml-org/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_0.gguf"


def fresh_root(path: pathlib.Path, chat_model: str = CHAT_MODEL) -> pathlib.Path:
    """A clean data root pinned to one chat model.

    Parameterised because the two crawl reels share this and must not share a model:
    running every reel on the same model makes the set look like one demo recorded
    several times.
    """
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    (path / "config.toml").write_text(
        f'chat_model = "{chat_model}"\n'
        'embedding_model = '
        '"nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"\n'
        'theme = "rose-pine"\n'
        "top_k = 8\n")
    (path / "data/lancedb").mkdir(parents=True)
    return path


def record(cast: pathlib.Path) -> dict:
    root = fresh_root(ROOT)

    s = drive.Session("reel-crawl", COLS, ROWS, cast)
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

        # 1. Open the modal. The URL field takes focus on its own.
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

        # 2. One page, fetched over plain HTTP. Both boxes start ticked.
        s.key("Tab", after=0.6)
        s.key("Space", after=0.8)
        s.key("Tab", after=0.6)
        s.key("Space", after=0.8)
        s.key("Tab", after=0.6)
        s.key("C-u", after=0.3)
        s.type_text("1", rate=0.05)
        time.sleep(0.4)
        s.key("Tab", after=0.6)
        s.key("C-u", after=0.3)
        s.type_text("0", rate=0.05)
        time.sleep(0.8)

        # 3. Fire it.
        s.key("Tab", after=0.6)
        s.key("Enter", after=1.0)
        s.wait_for(r"Recursive", absent=True, timeout=20)
        time.sleep(0.8)

        # 4. The crawl lands in the Task Center like any other ingest.
        s.esc()
        time.sleep(0.3)
        s.key("t", after=0.8)
        s.wait_for(r"Background Tasks", timeout=30)
        timings["crawl"] = s.wait_for(r"(crawl|add)\s+(done|complete)|all caught up",
                                      timeout=900)
        time.sleep(2.0)

        # 5. Ask something only that page can answer.
        s.goto("Chat", forward=False, limit=8, marker=r"Slash commands")
        time.sleep(0.6)
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
        time.sleep(3.5)

        timings["total"] = time.monotonic() - t0
        s.mark("payload_end")
        time.sleep(1.0)
    finally:
        s.kill()
    timings["marks"] = dict(s.marks)
    timings["motion_spans"] = list(s.motion_spans)
    return timings


if __name__ == "__main__":
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tui-crawl.cast")))
