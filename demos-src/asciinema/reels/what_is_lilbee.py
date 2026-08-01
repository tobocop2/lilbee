#!/usr/bin/env python3
"""what_is_lilbee: the README hero. Add a document, then ask it a question.

The arc is the demo: `/add` on camera with the path autocomplete showing, the Task Center
filling, and then an answer that cites the file that was just added. A version that hid
the add and showed only the question was recorded once; it is not the reference, because
without the add the answer is indistinguishable from a model talking from memory.

The document is lilbee's own README, so the answer is checkable by anyone watching.
"""
from __future__ import annotations

import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "what_is_lilbee"
COLS, ROWS = 128, 41
MUST_STRINGS = ("Background Tasks", "what is lilbee in one sentence", "README.md")

ROOT = pathlib.Path.home() / ".cache/lilbee-reel/whatis"
DOC = pathlib.Path.home() / "projects/lilbee/README.md"
QUESTION = "what is lilbee in one sentence?"

# 4B rather than 8B: it loads fast enough that the engine bar does not become the reel,
# and the answer is a single sentence off a README either way.
CONFIG = """chat_model = "Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"
embedding_model = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
theme = "rose-pine"
top_k = 8
"""


def record(cast: pathlib.Path) -> dict:
    shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True)
    (ROOT / "config.toml").write_text(CONFIG)
    # Empty, not absent: a missing lancedb directory re-triggers the setup wizard.
    (ROOT / "data/lancedb").mkdir(parents=True)

    s = drive.Session("reel-whatis", COLS, ROWS, cast)
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    s.start("lilbee", env={"LILBEE_DATA": str(ROOT)})
    try:
        timings["boot"] = s.wait_for(r"personal encyclopedia", timeout=120)
        time.sleep(1.2)
        s.repaint()
        s.wait_for(r"personal encyclopedia", timeout=20)
        time.sleep(0.6)
        s.mark("boot_end")

        # 1. Add the README, typed slowly enough that the path completion is readable.
        s.key("i", after=0.5)
        s.wait_for(r"INSERT", timeout=10)
        s.key("C-u", after=0.3)
        s.type_text(f"/add {DOC}", rate=0.055)
        time.sleep(1.6)
        s.key("Enter", after=1.0)
        s.wait_for(r"Add README\.md", timeout=30)
        s.esc()
        time.sleep(0.4)

        # 2. The Task Center. One README indexes in well under a second, so the row may
        # already have finished by the time the screen opens; accept either the finished
        # row or the idle state rather than waiting out a row that will never appear.
        s.key("t", after=0.8)
        s.wait_for(r"Background Tasks", timeout=30)
        timings["ingest"] = s.wait_for(r"add\s+(done|complete)|all caught up", timeout=600)
        time.sleep(1.6)

        # 3. Back to chat and ask.
        s.goto("Chat", forward=False, limit=8, marker=r"personal encyclopedia")
        time.sleep(0.6)
        s.key("i", after=0.5)
        s.wait_for(r"INSERT", timeout=10)
        s.type_text(QUESTION, rate=0.02)
        time.sleep(1.0)
        s.key("Enter", after=0.8)

        # 4. The answer, with its citation. Engine load is part of the wait on a cold
        # start, which is why the budget is generous rather than tuned to a warm run.
        timings["answer"] = s.wait_for(r"README\.md", timeout=600)
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
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/what_is_lilbee.cast")))
