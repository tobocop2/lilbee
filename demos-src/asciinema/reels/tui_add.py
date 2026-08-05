#!/usr/bin/env python3
"""tui-add: empty library to a cited answer, in one take.

The whole arc with nothing pre-staged: a real PDF goes in, the Task Center shows it being
extracted, chunked and embedded, and then the manual answers a question about itself with
a page number. what_is_lilbee does the same shape with a README that indexes instantly;
this one uses a document big enough that the work is visible.

Per earlier direction the take lands on the Task Center rather than on Status, so the
last thing on screen is the ingest that made the answer possible.
"""
from __future__ import annotations

import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "tui-add"
COLS, ROWS = 128, 41
MUST_STRINGS = ("Background Tasks", "jump starting", "Sources:")
BEATS = (
    ("the add typed", r"/add "),
    ("ingest running", r"add\s+active|Syncing|\d+%"),
    ("ingest finished", r"add\s+(done|complete)|all caught up"),
    ("cited answer", r"Sources:"),
)

TAIL_FORBID = ("Cancel stream",)
SPEED_WINDOWS = ("ingest", "gen")

ROOT = pathlib.Path.home() / ".cache/lilbee-reel/add"
DOC = pathlib.Path.home() / "Downloads/cv-manual.pdf"
QUESTION = "what does the manual say about jump starting a dead battery?"

# Gemma 4 E4B: a different family from the Qwen reels, and fast enough that the answer
# does not dominate a reel whose subject is the ingest.
CONFIG = """chat_model = "ggml-org/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_0.gguf"
embedding_model = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
theme = "rose-pine"
top_k = 8
"""


def record(cast: pathlib.Path) -> dict:
    if not DOC.exists():
        raise SystemExit(f"missing {DOC}")
    shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True)
    (ROOT / "config.toml").write_text(CONFIG)
    (ROOT / "data/lancedb").mkdir(parents=True)
    incoming = ROOT / "incoming"
    incoming.mkdir()
    shutil.copy(DOC, incoming / DOC.name)

    s = drive.Session("reel-add", COLS, ROWS, cast)
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    s.start("lilbee", env={"LILBEE_DATA": str(ROOT)})
    try:
        timings["boot"] = s.await_chat(timeout=180)
        time.sleep(1.2)
        s.repaint()
        s.wait_for(r"personal encyclopedia|Slash commands", timeout=20)
        time.sleep(0.6)
        s.mark("boot_end")

        # 1. The add, typed deliberately so the path is readable.
        s.insert()
        s.type_text(f"/add {incoming}", rate=0.055)
        time.sleep(1.2)
        s.key("Enter", after=1.0)
        s.esc()
        time.sleep(0.4)

        # 2. The Task Center doing the work.
        s.key("t", after=0.8)
        s.wait_for(r"Background Tasks", timeout=30)
        s.mark("ingest_start")
        timings["ingest"] = s.await_task("add", finish=1800)
        s.mark("ingest_end")
        time.sleep(2.0)

        # 3. Ask the document about itself.
        s.goto("Chat", forward=False, limit=8, marker=r"Slash commands")
        time.sleep(0.6)
        s.key("C-g", after=1.0)
        s.wait_for(r"Placement", timeout=25)
        time.sleep(1.2)
        s.ask(QUESTION, rate=0.05)
        s.mark("gen_start")
        timings["answer"] = s.await_answer()
        s.mark("gen_end")
        time.sleep(3.0)

        # 4. End on the Task Center, which is what made the answer possible.
        s.esc()
        time.sleep(0.3)
        s.key("t", after=0.8)
        s.wait_for(r"Background Tasks", timeout=25)
        time.sleep(2.4)

        timings["total"] = time.monotonic() - t0
        s.mark("payload_end")
        time.sleep(1.0)
    finally:
        s.kill()
    timings["marks"] = dict(s.marks)
    timings["motion_spans"] = list(s.motion_spans)
    return timings


if __name__ == "__main__":
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tui-add.cast")))
