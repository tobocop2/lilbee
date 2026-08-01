#!/usr/bin/env python3
"""tui-ollama-document: lilbee answering from a manual through a model Ollama serves.

The point is provenance. Everything else in the reel set runs a model lilbee downloaded
and launched itself; here the chat model lives in Ollama and lilbee talks to it over
``ollama/``, while retrieval, ingest and the embedder stay local to lilbee. The header
names the model and the placement drawer shows the card, so both halves are on screen.

The document is a real 2.3 MB PDF rather than a text file, so the ingest is the ingest a
viewer would actually have: extraction, chunking and embedding, not a one-second add.

The question targets prose rather than a spec table on purpose. Asked about coolant
capacity, the answer was correct and cited page 278, but it quoted the retrieved chunk
back verbatim -- and this manual's tables extract as scrambled word salad without OCR, so
a correct answer read as a broken one. What a demo shows is not only whether the answer
is right.
"""
from __future__ import annotations

import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "tui-ollama-document"
COLS, ROWS = 128, 41
MUST_STRINGS = ("ollama", "Background Tasks", "Sources:")
TAIL_FORBID = ("Cancel stream",)
# The PDF ingest runs for minutes on this machine and is a progress bar throughout,
# so it is compressed like the generation window and labelled the same way.
SPEED_WINDOWS = ("ingest", "gen")

ROOT = pathlib.Path.home() / ".cache/lilbee-reel/ollama"
DOC = pathlib.Path.home() / "Downloads/cv-manual.pdf"
QUESTION = "what should I do if the engine overheats?"

# Served by Ollama, not by lilbee's own engine. The embedder stays local: the reel is
# about where the chat model lives, and swapping the embedder as well would only make
# the ingest slower without saying anything new.
CONFIG = """chat_model = "ollama/qwen3:4b"
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

    s = drive.Session("reel-ollama", COLS, ROWS, cast)
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    s.start("lilbee", env={"LILBEE_DATA": str(ROOT)})
    try:
        timings["boot"] = s.wait_for(r"personal encyclopedia", timeout=180)
        time.sleep(1.2)
        s.repaint()
        s.wait_for(r"personal encyclopedia", timeout=20)
        time.sleep(0.6)
        s.mark("boot_end")

        # 1. Where the model comes from. /models lists what lilbee can reach, and the
        # only chat entry is the one Ollama is serving.
        s.key("i", after=0.5)
        s.key("C-u", after=0.3)
        s.type_text("/models", rate=0.05)
        time.sleep(0.6)
        s.key("Enter", after=1.0)
        time.sleep(2.8)
        s.esc()
        time.sleep(0.6)

        # 2. The manual, added on camera.
        s.key("i", after=0.5)
        s.key("C-u", after=0.3)
        s.type_text(f"/add {incoming}", rate=0.05)
        time.sleep(1.0)
        s.key("Enter", after=1.0)
        s.esc()
        time.sleep(0.4)
        s.key("t", after=0.8)
        s.wait_for(r"Background Tasks", timeout=30)
        s.mark("ingest_start")
        timings["ingest"] = s.wait_for(r"add\s+(done|complete)|all caught up", timeout=1200)
        s.mark("ingest_end")
        time.sleep(1.8)

        # 3. Ask it, with the placement drawer open.
        s.goto("Chat", forward=False, limit=8, marker=r"Slash commands")
        time.sleep(0.6)
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
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                              else "/tmp/tui-ollama-document.cast")))
